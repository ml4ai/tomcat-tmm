#include "CPD.h"

#include "pgm/RandomVariableNode.h"

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

        Eigen::MatrixXd CPD::sample(const shared_ptr<gsl_rng>& random_generator,
                                    const vector<shared_ptr<Node>>& index_nodes,
                                    int num_samples) const {

            vector<int> distribution_indices =
                this->get_indexed_distribution_indices(index_nodes,
                                                       num_samples);

            int sample_size = this->distributions[0]->get_sample_size();

            Eigen::MatrixXd samples(distribution_indices.size(), sample_size);
            for (int i = 0; i < num_samples; i++) {
                int distribution_idx = distribution_indices[i];
                Eigen::VectorXd sample =
                    this->distributions[distribution_idx]->sample(
                        random_generator, i);
                samples.row(i) = move(sample);
            }

            return samples;
        }

        Eigen::MatrixXd CPD::sample_from_posterior(
            const shared_ptr<gsl_rng>& random_generator,
            const vector<shared_ptr<Node>>& index_nodes,
            const Eigen::MatrixXd& posterior_weights) const {

            int num_indices = posterior_weights.rows();
            vector<int> distribution_indices =
                this->get_indexed_distribution_indices(index_nodes,
                                                       num_indices);

            int cols = this->distributions[0]->get_sample_size();
            Eigen::MatrixXd sample(num_indices, cols);
            for (int i = 0; i < num_indices; i++) {
                int distribution_idx = distribution_indices[i];
                const shared_ptr<Distribution>& distribution =
                    this->distributions[distribution_idx];
                sample.row(i) = distribution->sample(random_generator,
                                                     posterior_weights.row(i));
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
                    // up to a
                    // certain row (num_indices).
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
            const Eigen::MatrixXd& cpd_owner_assignment) const {

            int rows = cpd_owner_assignment.rows();
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
                    shared_ptr<Distribution> distribution =
                        this->distributions[distribution_idx + j * offset];

                    weights(i, j) =
                        distribution->get_pdf(cpd_owner_assignment.row(i));
                }
            }

            return weights;
        }

        void CPD::update_sufficient_statistics(
            const vector<shared_ptr<Node>>& index_nodes,
            const Eigen::MatrixXd& cpd_owner_assignment) {

            int n = cpd_owner_assignment.rows();
            vector<int> distribution_indices =
                this->get_indexed_distribution_indices(index_nodes, n);
            unordered_map<int, vector<double>> values_per_distribution;

            for (int i = 0; i < n; i++) { // O(d)
                int distribution_idx = distribution_indices[i];
                double value = cpd_owner_assignment(i, 0);

                if (!EXISTS(distribution_idx, values_per_distribution)) {
                    values_per_distribution[distribution_idx] = {};
                }

                values_per_distribution[distribution_idx].push_back(value);
            }

            for (auto& [distribution_idx, values] :
                 values_per_distribution) { // O(d)
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
