#include "pgm/cpd/CategoricalCPD.h"
#include "pgm/ConstantNode.h"

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

            Eigen::MatrixXd distributions_table = this->get_table(0);

            int num_distributions = this->distributions.size();
            Eigen::MatrixXd binary_distribution_indices =
                Eigen::MatrixXd::Zero(rows, num_distributions);
            Eigen::MatrixXd binary_assignment =
                Eigen::MatrixXd::Zero(rows, distributions_table.cols());

            for (int i = 0; i < rows; i++) {
                binary_assignment(i, cpd_owner_assignment(i, 0)) = 1;
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

    } // namespace model
} // namespace tomcat
