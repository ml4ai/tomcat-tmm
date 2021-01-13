#include "pgm/cpd/CategoricalCPD.h"

#include "pgm/ConstantNode.h"
#include "pgm/TimerNode.h"

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        CategoricalCPD::CategoricalCPD(
            const vector<shared_ptr<NodeMetadata>>& parent_node_order,
            const vector<shared_ptr<Categorical>>& distributions)
            : CPD(parent_node_order) {
            this->init_from_distributions(distributions);
        }

        CategoricalCPD::CategoricalCPD(
            vector<shared_ptr<NodeMetadata>>&& parent_node_order,
            const vector<shared_ptr<Categorical>>& distributions)
            : CPD(parent_node_order) {
            this->init_from_distributions(distributions);
        }

        CategoricalCPD::CategoricalCPD(
            const vector<shared_ptr<NodeMetadata>>& parent_node_order,
            const Eigen::MatrixXd& probabilities)
            : CPD(parent_node_order) {
            this->init_from_matrix(probabilities);
        }

        CategoricalCPD::CategoricalCPD(
            vector<shared_ptr<NodeMetadata>>&& parent_node_order,
            const Eigen::MatrixXd& cpd_table)
            : CPD(parent_node_order) {
            this->init_from_matrix(cpd_table);
        }

        CategoricalCPD::~CategoricalCPD() {}

        //----------------------------------------------------------------------
        // Copy & Move constructors/assignments
        //----------------------------------------------------------------------
        CategoricalCPD::CategoricalCPD(const CategoricalCPD& cpd) {
            this->copy_cpd(cpd);
        }

        CategoricalCPD& CategoricalCPD::operator=(const CategoricalCPD& cpd) {
            this->copy_cpd(cpd);
            return *this;
        }

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        void CategoricalCPD::init_from_distributions(
            const vector<shared_ptr<Categorical>>& distributions) {
            this->distributions.reserve(distributions.size());
            for (const auto& distribution : distributions) {
                this->distributions.push_back(distribution);
            }
        }

        void CategoricalCPD::init_from_matrix(const Eigen::MatrixXd& matrix) {
            for (int i = 0; i < matrix.rows(); i++) {
                shared_ptr<Categorical> distribution_ptr =
                    make_shared<Categorical>(Categorical(move(matrix.row(i))));
                this->distributions.push_back(distribution_ptr);
            }
        }

        unique_ptr<CPD> CategoricalCPD::clone() const {
            unique_ptr<CategoricalCPD> new_cpd =
                make_unique<CategoricalCPD>(*this);
            new_cpd->clone_distributions();
            return new_cpd;
        }

        void CategoricalCPD::clone_distributions() {
            for (auto& distribution : this->distributions) {
                shared_ptr<Distribution> temp = distribution->clone();
                distribution = dynamic_pointer_cast<Categorical>(temp);
            }
        }

        string CategoricalCPD::get_description() const {
            stringstream ss;
            ss << "Categorical CPD: {\n";
            for (auto& probabilities : this->distributions) {
                ss << " " << *probabilities << "\n";
            }
            ss << "}";

            return ss.str();
        }

        void CategoricalCPD::add_to_sufficient_statistics(
            const vector<double>& values) {
            throw invalid_argument(
                "No conjugate prior with a categorical distribution.");
        }

        Eigen::MatrixXd CategoricalCPD::sample_from_conjugacy(
            const shared_ptr<gsl_rng>& random_generator,
            const vector<shared_ptr<Node>>& parent_nodes,
            int num_samples) const {
            throw invalid_argument(
                "No conjugate prior with a categorical distribution.");
        }

        void CategoricalCPD::reset_sufficient_statistics() {
            // Nothing to reset
        }

        Eigen::MatrixXd CategoricalCPD::get_posterior_weights(
            const vector<shared_ptr<Node>>& index_nodes,
            const shared_ptr<RandomVariableNode>& sampled_node,
            const std::shared_ptr<const RandomVariableNode>& cpd_owner) const {

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

            Eigen::MatrixXd distributions_table = this->get_table(0);
            if (sampled_node->has_timer() &&
                cpd_owner->get_previous() == sampled_node) {
                // We ignore the probability of staying in the same state as
                // this will be embedded in the segment posterior weights.
                distributions_table.diagonal() =
                    Eigen::VectorXd::Ones(distributions_table.rows());
            }

            int num_distributions = this->distributions.size();
            Eigen::MatrixXd binary_distribution_indices =
                Eigen::MatrixXd::Zero(rows, num_distributions);
            Eigen::MatrixXd binary_assignment =
                Eigen::MatrixXd::Zero(rows, distributions_table.cols());

            for (int i = 0; i < rows; i++) {
                binary_assignment(i, cpd_owner->get_assignment()(i, 0)) = 1;
            }

            Eigen::MatrixXd weights(rows, cols);
            const string& sampled_node_label =
                sampled_node->get_metadata()->get_label();
            // For every possible value of sampled_node, the offset indicates
            // how many distributions ahead we need to advance to get the
            // distribution indexes by the index nodes of this CPD.
            int offset = this->parent_label_to_indexing.at(sampled_node_label)
                             .right_cumulative_cardinality;
            for (int j = 0; j < cols; j++) {
                for (int i = 0; i < rows; i++) {
                    int distribution_idx = distribution_indices[i];
                    if (j > 0) {
                        binary_distribution_indices(
                            i, distribution_idx + (j - 1) * offset) = 0;
                    }
                    binary_distribution_indices(
                        i, distribution_idx + j * offset) = 1;
                }

                weights.col(j) =
                    ((binary_distribution_indices * distributions_table)
                         .array() *
                     binary_assignment.array())
                        .rowwise()
                        .sum();
            }

            return weights;
        }

        Eigen::MatrixXd CategoricalCPD::get_segment_posterior_weights(
            const vector<shared_ptr<Node>>& index_nodes,
            const shared_ptr<RandomVariableNode>& sampled_node,
            const std::shared_ptr<const RandomVariableNode>& cpd_owner) const {

            const auto& previous = sampled_node->get_previous();
            const auto& next = sampled_node->get_next();
            const auto& previous_timer =
                dynamic_pointer_cast<TimerNode>(cpd_owner->get_previous());
            const auto& next_timer =
                dynamic_pointer_cast<TimerNode>(cpd_owner->get_next());

            const Eigen::VectorXd prev_assignments =
                previous ? previous->get_assignment().col(0)
                         : sampled_node->get_assignment().col(0);
            const Eigen::VectorXd next_assignments =
                next ? next->get_assignment().col(0)
                     : sampled_node->get_assignment().col(0);

            int rows = cpd_owner->get_size();
            Eigen::VectorXd prev_durations;
            if (previous_timer) {
                prev_durations = previous_timer->get_assignment().col(0);
            }
            else {
                prev_durations = Eigen::VectorXd::Ones(rows);
            }
            Eigen::VectorXd next_durations;
            if (next_timer) {
                next_durations = next_timer->get_backward_assignment().col(0);
            }
            else {
                next_durations = Eigen::VectorXd::Ones(rows);
            }

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

            int num_distributions = this->distributions.size();

            Eigen::MatrixXd weights(rows, cols);
            const string& sampled_node_label =
                sampled_node->get_metadata()->get_label();
            // For every possible value of sampled_node, the offset indicates
            // how many distributions ahead we need to advance to get the
            // distribution indexes by the index nodes of this CPD.
            int offset = this->parent_label_to_indexing.at(sampled_node_label)
                             .right_cumulative_cardinality;

            for (int j = 0; j < cols; j++) {
                for (int i = 0; i < rows; i++) {
                    // Fix to get the distribution index at the beginning of
                    // the segment, which changes from row to row in the left
                    // segment.
                    int distribution_idx =
                        distribution_indices[i] + (j - 1) * offset;
                    const auto& timer_distribution =
                        this->distributions[distribution_idx];

                    int prev_duration = prev_durations(i);
                    int next_duration = next_durations(i);
                    int d = 0;

                    if (cpd_owner ==
                        sampled_node->get_timer()->get_previous()) {

                        if (prev_assignments(i) == j) {
                            if (next_assignments(i) == j) {
                                // Left and right segment, and sampled node
                                // form a unique segment.
                                d = prev_duration + next_duration + 1;
                            }
                            else {
                                // Left segment and sampled node form a
                                // unique segment. Right segment is different.
                                d = prev_duration + 1;
                            }
                        }
                        else {
                            // Left segment does not form a segment with the
                            // sampled node.
                            d = prev_duration;
                        }
                    }
                    else if (cpd_owner == sampled_node->get_timer()) {
                        if (next_assignments(i) != j && prev_assignments(i) != j) {
                            // Sampled node does not form a unique segment
                            // with either its left or right segment.
                            d = 1;
                        }
                        else {
                            // Any other probability configuration will be
                            // included in the weights of the left and right
                            // segments, whenever this method is called with
                            // each of them as cpd_owners.
                            weights(i, j) = 1;
                            continue;
                        }
                    }
                    else if (cpd_owner ==
                             sampled_node->get_timer()->get_next()) {
                        if (next_assignments(i) == j) {
                            if (prev_assignments(i) == j) {
                                // Left and right segment, and sampled node
                                // form a unique segment.
                                weights(i, j) = 1;
                                continue;
                            }
                            else {
                                // Left segment is different.
                                d = next_duration + 1;
                            }
                        }
                        else {
                            d = next_duration;
                        }
                    }
                    else {
                        stringstream ss;
                        ss << "The owner of the CPD " << this
                           << " is not any of the immediate timers of "
                           << sampled_node;
                        throw TomcatModelException(ss.str());
                    }

                    weights(i, j) = timer_distribution->get_pdf(
                        Eigen::VectorXd::Constant(1, d));
                }
            }

            return weights;
        }

    } // namespace model
} // namespace tomcat
