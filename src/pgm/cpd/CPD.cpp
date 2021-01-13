#include "CPD.h"

#include "pgm/RandomVariableNode.h"
#include "pgm/TimerNode.h"

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        CPD::CPD() {}

        CPD::CPD(const vector<shared_ptr<NodeMetadata>>& parent_node_order)
            : parent_node_order(parent_node_order) {
            this->init_id();
            this->fill_indexing_mapping();
            this->sufficient_statistics_mutex = make_unique<mutex>();
        }

        CPD::CPD(vector<shared_ptr<NodeMetadata>>&& parent_node_order)
            : parent_node_order(move(parent_node_order)) {
            this->init_id();
            this->fill_indexing_mapping();
            this->sufficient_statistics_mutex = make_unique<mutex>();
        }

        CPD::CPD(const vector<shared_ptr<NodeMetadata>>& parent_node_order,
                 const vector<shared_ptr<Distribution>>& distributions)
            : parent_node_order(parent_node_order),
              distributions(distributions) {
            this->init_id();
            this->fill_indexing_mapping();
            this->sufficient_statistics_mutex = make_unique<mutex>();
        }

        CPD::CPD(vector<shared_ptr<NodeMetadata>>&& parent_node_order,
                 vector<shared_ptr<Distribution>>&& distributions)
            : parent_node_order(move(parent_node_order)),
              distributions(move(distributions)) {
            this->init_id();
            this->fill_indexing_mapping();
            this->sufficient_statistics_mutex = make_unique<mutex>();
        }

        CPD::~CPD() {}

        //----------------------------------------------------------------------
        // Operator overload
        //----------------------------------------------------------------------
        ostream& operator<<(ostream& os, const CPD& cpd) {
            cpd.print(os);
            return os;
        }

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        void CPD::init_id() {
            vector<string> labels;
            labels.reserve(this->parent_node_order.size());

            for (const auto& metadata : this->parent_node_order) {
                labels.push_back(metadata->get_label());
            }

            sort(labels.begin(), labels.end());
            stringstream ss;
            copy(labels.begin(),
                 labels.end(),
                 ostream_iterator<string>(ss, ","));
            this->id = ss.str();
        }

        void CPD::fill_indexing_mapping() {
            int cum_cardinality = 1;

            for (int order = this->parent_node_order.size() - 1; order >= 0;
                 order--) {
                shared_ptr<NodeMetadata> metadata =
                    this->parent_node_order[order];

                ParentIndexing indexing(
                    order, metadata->get_cardinality(), cum_cardinality);
                this->parent_label_to_indexing[metadata->get_label()] =
                    indexing;

                cum_cardinality *= metadata->get_cardinality();
            }
        }

        void CPD::copy_cpd(const CPD& cpd) {
            this->id = cpd.id;
            this->updated = cpd.updated;
            this->parent_label_to_indexing = cpd.parent_label_to_indexing;
            this->parent_node_order = cpd.parent_node_order;
            this->distributions = cpd.distributions;
            this->sufficient_statistics_mutex = make_unique<mutex>();
        }

        void CPD::update_dependencies(const Node::NodeMap& parameter_nodes_map,
                                      int time_step) {

            for (auto& distribution : this->distributions) {
                distribution->update_dependencies(parameter_nodes_map,
                                                  time_step);
            }

            this->updated = true;
        }

        Eigen::MatrixXd
        CPD::sample(const shared_ptr<gsl_rng>& random_generator,
                    const vector<shared_ptr<Node>>& index_nodes,
                    int num_samples,
                    const std::shared_ptr<const RandomVariableNode>&
                        sampled_node) const {

            vector<int> distribution_indices =
                this->get_indexed_distribution_indices(index_nodes,
                                                       num_samples);

            int sample_size = this->distributions[0]->get_sample_size();

            const auto& timer = sampled_node->get_timer();
            const auto& previous = sampled_node->get_previous();

            Eigen::MatrixXd samples(distribution_indices.size(), sample_size);
            for (int i = 0; i < num_samples; i++) {
                int distribution_idx = distribution_indices[i];

                Eigen::VectorXd sample;

                if (sampled_node->get_metadata()->is_timer()) {
                    bool decrement_sample = false;

                    if (previous) {
                        if (previous->get_assignment()(i, 0) > 0) {
                            decrement_sample = true;
                        }
                    }

                    if (decrement_sample) {
                        sample = previous->get_assignment().row(i).array() - 1;
                    }
                    else {
                        sample = this->distributions[distribution_idx]->sample(
                            random_generator, i);
                    }
                }
                else {
                    bool repeat_sample = false;

                    if (sampled_node->get_timer()) {
                        const auto& previous_timer =
                            dynamic_pointer_cast<RandomVariableNode>(
                                sampled_node->get_timer())
                                ->get_previous();
                        if (previous_timer) {
                            if (previous_timer->get_assignment()(i, 0) > 0) {
                                // Node is controlled by a timer and previous
                                // count value did not reach 0 yet.
                                repeat_sample = true;
                            }
                        }
                    }

                    if (repeat_sample) {
                        sample = previous->get_assignment().row(i);
                    }
                    else {
                        sample = this->distributions[distribution_idx]->sample(
                            random_generator, i);
                    }
                }

                samples.row(i) = move(sample);
            }

            return samples;
        }

        Eigen::MatrixXd CPD::sample_from_posterior(
            const shared_ptr<gsl_rng>& random_generator,
            const vector<shared_ptr<Node>>& index_nodes,
            const Eigen::MatrixXd& posterior_weights,
            const std::shared_ptr<const RandomVariableNode>& cpd_owner) const {

            int num_indices = posterior_weights.rows();
            vector<int> distribution_indices =
                this->get_indexed_distribution_indices(index_nodes,
                                                       num_indices);
            const auto& previous = cpd_owner->get_previous();

            int cols = this->distributions[0]->get_sample_size();
            Eigen::MatrixXd sample(num_indices, cols);
            for (int i = 0; i < num_indices; i++) {
                int distribution_idx = distribution_indices[i];
                const auto& distribution =
                    this->distributions[distribution_idx];
                const auto& weights = posterior_weights.row(i);

                if (cpd_owner->has_timer() && previous) {
                    // If the node that owns this CPD is controlled by a timer,
                    // P(node|Parents(node)) is calculated according to its left
                    // and right segments. If node(t-1) == node(t), we
                    // must ignore the transition probability from the
                    // transition matrix as there's no transition to a
                    // different state. This would affect only the duration of
                    // the previous segment (or the full segment if node(t+1)
                    // == node(t) as well) which will be increased by one. To
                    // avoid returning 0 (or whatever value defined in the
                    // transition matrix when node(t) == node(t-1)) we pass the
                    // previous state to the sampling method so that P(node
                    // (t)|node (t-1)) s.t. node(t-1) == node(t) equals 1. The
                    // actual probability of this "transition", regarding the
                    // duration of the left and right segments, is embedded in
                    // the weights passed to the sampling method.
                    double previous_value = previous->get_assignment()(i, 0);
                    sample.row(i) = distribution->sample(
                        random_generator, weights, previous_value);
                }
                else {
                    sample.row(i) =
                        distribution->sample(random_generator, weights);
                }
            }

            return sample;
        }

        vector<int> CPD::get_indexed_distribution_indices(
            const vector<shared_ptr<Node>>& index_nodes,
            int num_indices) const {

            Eigen::MatrixXd indices = Eigen::MatrixXd::Zero(num_indices, 1);
            for (auto& index_node : index_nodes) {
                const string& label = index_node->get_metadata()->get_label();
                const ParentIndexing& indexing =
                    this->parent_label_to_indexing.at(label);

                if (index_node->get_assignment().rows() > num_indices) {
                    // Ancestral sampling may generate less samples for a
                    // child than the number of samples for its parents in
                    // the case of homogeneous world, for instance. This
                    // only considers samples of the parent nodes (index nodes)
                    // up to a certain row (num_indices).
                    indices = indices.array() +
                              index_node->get_assignment()
                                      .block(0, 0, num_indices, 1)
                                      .array() *
                                  indexing.right_cumulative_cardinality;
                }
                else {
                    indices = indices.array() +
                              index_node->get_assignment().array() *
                                  indexing.right_cumulative_cardinality;
                }
            }

            vector<int> vec(indices.data(), indices.data() + num_indices);

            return vec;
        }

        Eigen::MatrixXd CPD::get_posterior_weights(
            const vector<shared_ptr<Node>>& index_nodes,
            const shared_ptr<RandomVariableNode>& sampled_node,
            const shared_ptr<const RandomVariableNode>& cpd_owner) const {

            if (sampled_node->has_timer() &&
                cpd_owner->get_previous() == sampled_node) {
                throw TomcatModelException(
                    "This implementation only supports calculating the "
                    "posterior for nodes with categorical distribution, in "
                    "case they are controlled by a timer.");
            }

            int rows = cpd_owner->get_size();
            int cols = sampled_node->get_metadata()->get_cardinality();

            // Set sampled node's assignment equals to zero so we can get the
            // index of the first distribution indexed by this node and the
            // other parent nodes that the child (owner of this CPD) may have.
            Eigen::MatrixXd saved_assignment = sampled_node->get_assignment();
            sampled_node->set_assignment(Eigen::MatrixXd::Zero(rows, 1));
            vector<int> distribution_indices =
                this->get_indexed_distribution_indices(index_nodes, rows);
            // Restore the sampled node's assignment to its original state.
            sampled_node->set_assignment(saved_assignment);

            Eigen::MatrixXd weights(rows, cols);
            const string& sampled_node_label =
                sampled_node->get_metadata()->get_label();
            // For every possible value of sampled_node, the offset indicates
            // how many distributions ahead we need to advance to get the
            // distribution indexes by the index nodes of this CPD.
            int offset = this->parent_label_to_indexing.at(sampled_node_label)
                             .right_cumulative_cardinality;

            for (int i = 0; i < rows; i++) {
                int distribution_idx = distribution_indices[i];

                for (int j = 0; j < cols; j++) {
                    const auto& distribution =
                        this->distributions[distribution_idx + j * offset];
                    weights(i, j) = distribution->get_pdf(
                        cpd_owner->get_assignment().row(i));
                }
            }

            return weights;
        }

        Eigen::MatrixXd CPD::get_segment_posterior_weights(
            const vector<shared_ptr<Node>>& index_nodes,
            const shared_ptr<RandomVariableNode>& sampled_node,
            const shared_ptr<const RandomVariableNode>& cpd_owner) const {

            if (sampled_node->has_timer() &&
                cpd_owner->get_previous() == sampled_node) {
                throw TomcatModelException(
                    "This implementation only supports calculating the "
                    "posterior for nodes with categorical distribution, in "
                    "case they are controlled by a timer.");
            }

            return Eigen::MatrixXd(0, 0);
        }

        void CPD::update_sufficient_statistics(
            const vector<shared_ptr<Node>>& index_nodes,
            const shared_ptr<RandomVariableNode>& cpd_owner) {

            int n = cpd_owner->get_size();
            vector<int> distribution_indices =
                this->get_indexed_distribution_indices(index_nodes, n);
            unordered_map<int, vector<double>> values_per_distribution;

            const Eigen::MatrixXd& values = cpd_owner->get_assignment();

            for (int i = 0; i < n; i++) {
                int distribution_idx = distribution_indices[i];

                if (!EXISTS(distribution_idx, values_per_distribution)) {
                    values_per_distribution[distribution_idx] = {};
                }

                double value = 0;
                bool add_to_list = false;
                if (cpd_owner->get_metadata()->is_timer()) {
                    // Only add to the sufficient statistics, when the timer
                    // starts over which is the moment when a transition
                    // between different states occur.
                    if (values(i, 0) == 0 || !cpd_owner->get_next()) {
                        // Beginning of a new segment or end of the last segment
                        if (const auto& previous_timer =
                                cpd_owner->get_previous()) {
                            // Size of the previous segment or last one
                            value = previous_timer->get_assignment()(i, 0);
                        }
                        else {
                            // First segment
                            value = values(i, 0);
                        }
                        add_to_list = true;
                    }
                }
                else {
                    value = values(i, 0);
                    add_to_list = true;
                }

                if (add_to_list) {
                    values_per_distribution[distribution_idx].push_back(value);
                }
            }

            for (auto& [distribution_idx, values] :
                 values_per_distribution) {
                const shared_ptr<Distribution>& distribution =
                    this->distributions[distribution_idx];
                distribution->update_sufficient_statistics(values);
            }
        }

        void CPD::reset_updated_status() { this->updated = false; }

        void CPD::print(ostream& os) const { os << this->get_description(); }

        Eigen::MatrixXd CPD::get_table(int parameter_idx) const {
            Eigen::MatrixXd table;

            int row = 0;
            for (const auto& distribution : this->distributions) {
                Eigen::VectorXd parameters =
                    distribution->get_values(parameter_idx);
                if (table.size() == 0) {
                    table = Eigen::MatrixXd(this->distributions.size(),
                                            parameters.size());
                    table.row(row) = parameters;
                }
                else {
                    table.row(row) = parameters;
                }
                row++;
            }

            return table;
        }

        //------------------------------------------------------------------
        // Getters & Setters
        //------------------------------------------------------------------
        const string& CPD::get_id() const { return id; }

        bool CPD::is_updated() const { return updated; }

        const CPD::TableOrderingMap& CPD::get_parent_label_to_indexing() const {
            return parent_label_to_indexing;
        }

    } // namespace model
} // namespace tomcat
