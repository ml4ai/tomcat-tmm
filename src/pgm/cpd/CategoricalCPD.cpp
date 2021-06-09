#include "pgm/cpd/CategoricalCPD.h"

#include <thread>

#include "pgm/ConstantNode.h"
#include "pgm/TimerNode.h"
#include "utils/EigenExtensions.h"
#include "utils/Multithreading.h"

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

            this->freeze_distributions(0);
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

        string CategoricalCPD::get_name() const { return "Categorical"; }

        void CategoricalCPD::add_to_sufficient_statistics(
            const vector<double>& values) {
            throw invalid_argument(
                "No conjugate prior with a categorical distribution.");
        }

        Eigen::MatrixXd CategoricalCPD::sample_from_conjugacy(
            const shared_ptr<gsl_rng>& random_generator,
            int num_samples,
            const shared_ptr<const RandomVariableNode>& cpd_owner) const {
            throw invalid_argument(
                "No conjugate prior with a categorical distribution.");
        }

        void CategoricalCPD::reset_sufficient_statistics() {
            // Nothing to reset
        }

        Eigen::MatrixXd CategoricalCPD::get_posterior_weights(
            const vector<shared_ptr<Node>>& index_nodes,
            const shared_ptr<RandomVariableNode>& sampled_node,
            const std::shared_ptr<const RandomVariableNode>& cpd_owner,
            int num_jobs) const {

            int data_size = cpd_owner->get_size();
            int cardinality = sampled_node->get_metadata()->get_cardinality();

            // Set sampled node's assignment equals to zero so we can get the
            // index of the first distribution indexed by this node and the
            // other parent nodes that the child (owner of this CPD) may have.
            Eigen::MatrixXd saved_assignment = sampled_node->get_assignment();
            sampled_node->set_assignment(Eigen::MatrixXd::Zero(data_size, 1));
            Eigen::VectorXi distribution_indices =
                this->get_indexed_distribution_indices(index_nodes, data_size);
            // Restore the sampled node's assignment to its original state.
            sampled_node->set_assignment(saved_assignment);

            Eigen::MatrixXd distributions_table = this->get_table(0);
            //            if (sampled_node->has_timer() &&
            //                cpd_owner->get_previous() == sampled_node) {
            if (cpd_owner->get_previous() && cpd_owner->has_timer()) {
                // We ignore the probability of staying in the same state as
                // this will be embedded in the segment posterior weights.
                int cpd_owner_cardinality =
                    cpd_owner->get_metadata()->get_cardinality();
                if (cpd_owner_cardinality == distributions_table.rows()) {
                    // Node only depends on a past copy of itself
                    distributions_table.diagonal() =
                        Eigen::VectorXd::Ones(distributions_table.rows());
                }
                else {
                    // Node depends on a past copy of itself plus other nodes
                    // . When defining a CPD, this implementation requires
                    // that the replicable node is defined as the first one in
                    // the index node of the CPD, which guarantees that the
                    // following modifications will always change the
                    // elements of the table that represent p(node(t-1)
                    // == node(t)).
                    int right_cum_cardinality =
                        distributions_table.rows() / cpd_owner_cardinality;
                    for (int i = 0; i < cpd_owner_cardinality; i++) {
                        distributions_table.block(i * right_cum_cardinality,
                                                  i,
                                                  right_cum_cardinality,
                                                  1) =
                            Eigen::VectorXd::Ones(right_cum_cardinality);
                    }
                }
            }

            const string& sampled_node_label =
                sampled_node->get_metadata()->get_label();
            // For every possible value of sampled_node, the offset indicates
            // how many distributions ahead we need to advance to get the
            // distribution indexes by the index nodes of this CPD.
            int offset = this->parent_label_to_indexing.at(sampled_node_label)
                             .right_cumulative_cardinality;

            Eigen::MatrixXd weights =
                this->compute_posterior_weights(cpd_owner,
                                                distribution_indices,
                                                cardinality,
                                                offset,
                                                distributions_table,
                                                num_jobs);
            return weights;
        }

        Eigen::MatrixXd CategoricalCPD::compute_posterior_weights(
            const shared_ptr<const RandomVariableNode>& cpd_owner,
            const Eigen::VectorXi& distribution_indices,
            int cardinality,
            int distribution_index_offset,
            const Eigen::MatrixXd& distributions_table,
            int num_jobs) const {

            int data_size = cpd_owner->get_size();
            Eigen::MatrixXd weights(data_size, cardinality);
            mutex weights_mutex;

            if (num_jobs == 1) {
                // Run in the main thread
                this->run_posterior_weights_thread(cpd_owner,
                                                   distribution_indices,
                                                   distribution_index_offset,
                                                   distributions_table,
                                                   make_pair(0, data_size),
                                                   weights,
                                                   weights_mutex);
            }
            else {
                vector<thread> threads;
                const vector<pair<int, int>> processing_blocks =
                    get_parallel_processing_blocks(num_jobs, data_size);
                for (const auto& processing_block : processing_blocks) {
                    thread weights_thread(
                        &CategoricalCPD::run_posterior_weights_thread,
                        this,
                        cpd_owner,
                        distribution_indices,
                        distribution_index_offset,
                        distributions_table,
                        ref(processing_block),
                        ref(weights),
                        ref(weights_mutex));
                    threads.push_back(move(weights_thread));
                }

                for (auto& weights_thread : threads) {
                    weights_thread.join();
                }
            }

            return weights;
        }

        void CategoricalCPD::run_posterior_weights_thread(
            const shared_ptr<const RandomVariableNode>& cpd_owner,
            const Eigen::VectorXi& distribution_indices,
            int distribution_index_offset,
            const Eigen::MatrixXd& distributions_table,
            const pair<int, int>& processing_block,
            Eigen::MatrixXd& full_weights,
            mutex& weights_mutex) const {

            int initial_row = processing_block.first;
            int num_rows = processing_block.second;
            int cardinality = full_weights.cols();

            const Eigen::VectorXi& assignment =
                cpd_owner->get_assignment()
                    .block(initial_row, 0, num_rows, 1)
                    .cast<int>();
            Eigen::MatrixXi binary_assignment =
                to_categorical(assignment, distributions_table.cols());

            int num_distributions = distributions_table.rows();
            Eigen::MatrixXi binary_distribution_indices =
                Eigen::MatrixXi::Zero(num_rows, num_distributions);
            Eigen::MatrixXd weights(num_rows, cardinality);
            for (int j = 0; j < cardinality; j++) {
                // Get the index for the next value of the indexing node
                // in binary format.
                for (int i = initial_row; i < initial_row + num_rows; i++) {
                    int distribution_idx = distribution_indices[i];

                    if (j > 0) {
                        // Zero the previous j
                        int prev_val_idx = distribution_idx +
                                           (j - 1) * distribution_index_offset;
                        binary_distribution_indices(i - initial_row,
                                                    prev_val_idx) = 0;
                    }

                    int curr_val_idx =
                        distribution_idx + j * distribution_index_offset;
                    binary_distribution_indices(i - initial_row, curr_val_idx) =
                        1;
                }

                weights.col(j) = ((binary_distribution_indices.cast<double>() *
                                   distributions_table)
                                      .array() *
                                  binary_assignment.cast<double>().array())
                                     .rowwise()
                                     .sum();
            }

            scoped_lock lock(weights_mutex);
            full_weights.block(initial_row, 0, num_rows, cardinality) = weights;
        }

        shared_ptr<CPD>
        CategoricalCPD::create_from_data(const EvidenceSet& data,
                                         const string& cpd_owner_label,
                                         int cpd_owner_cardinality) {
            Eigen::MatrixXd prob_table = Eigen::MatrixXd::Zero(
                this->distributions.size(), cpd_owner_cardinality);

            for (int d = 0; d < data.get_num_data_points(); d++) {
                int distribution_idx = 0;
                for (const auto& [node_label, indexing_scheme] :
                     this->parent_label_to_indexing) {
                    if (!data.has_data_for(node_label)) {
                        stringstream ss;
                        ss << "There's no data for the index node "
                           << node_label << ".";
                        throw TomcatModelException(ss.str());
                    }

                    // The values of interest must be in the first column of
                    // the matrix that comprises the data tensor.
                    distribution_idx +=
                        data[node_label].at(0, d, 0) *
                        indexing_scheme.right_cumulative_cardinality;
                }

                int k = data[cpd_owner_label].at(0, d, 0);
                prob_table(distribution_idx, k) += 1;
            }

            Eigen::VectorXd sum_per_row =
                prob_table.rowwise().sum().array() + EPSILON;
            prob_table = prob_table.array().colwise() / sum_per_row.array();

            return make_shared<CategoricalCPD>(this->parent_node_order,
                                               prob_table);
        }

        bool CategoricalCPD::is_continuous() const { return false; }

        Eigen::VectorXd CategoricalCPD::get_pdfs(
            const shared_ptr<const RandomVariableNode>& cpd_owner,
            int num_jobs,
            int parameter_idx) const {

            int num_samples = cpd_owner->get_size();
            Eigen::VectorXi distribution_indices =
                this->get_indexed_distribution_indices(cpd_owner->get_parents(),
                                                       num_samples);
            Eigen::MatrixXd cpd_table = this->get_table(parameter_idx);

            Eigen::VectorXd pdfs(num_samples);
            mutex pdfs_mutex;
            if (num_jobs == 1) {
                // Run in the main thread
                this->run_pdf_thread(cpd_owner,
                                     distribution_indices,
                                     cpd_table,
                                     parameter_idx,
                                     pdfs,
                                     make_pair(0, num_samples),
                                     pdfs_mutex);
            }
            else {
                const auto processing_blocks =
                    get_parallel_processing_blocks(num_jobs, num_samples);
                vector<thread> threads;
                for (int i = 0; i < processing_blocks.size(); i++) {
                    thread pdfs_thread(&CategoricalCPD::run_pdf_thread,
                                       this,
                                       ref(cpd_owner),
                                       ref(distribution_indices),
                                       ref(cpd_table),
                                       parameter_idx,
                                       ref(pdfs),
                                       ref(processing_blocks.at(i)),
                                       ref(pdfs_mutex));
                    threads.push_back(move(pdfs_thread));
                }

                for (auto& samples_thread : threads) {
                    samples_thread.join();
                }
            }

            return pdfs;
        }

        void CategoricalCPD::run_pdf_thread(
            const shared_ptr<const RandomVariableNode>& cpd_owner,
            const Eigen::VectorXi& distribution_indices,
            const Eigen::MatrixXd& cpd_table,
            int parameter_idx,
            Eigen::VectorXd& full_pdfs,
            const pair<int, int>& processing_block,
            mutex& pdf_mutex) const {

            int initial_row = processing_block.first;
            int num_rows = processing_block.second;

            Eigen::VectorXd pdfs(num_rows, 1);
            for (int i = initial_row; i < initial_row + num_rows; i++) {
                pdfs[i - initial_row] = cpd_table(
                    distribution_indices[i], cpd_owner->get_assignment()(i, 0));
            }

            scoped_lock lock(pdf_mutex);
            full_pdfs.block(initial_row, 0, num_rows, 1) = pdfs;
        }

    } // namespace model
} // namespace tomcat
