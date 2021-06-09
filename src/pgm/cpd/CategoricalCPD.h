#pragma once

#include "distribution/Categorical.h"
#include "pgm/RandomVariableNode.h"
#include "pgm/cpd/CPD.h"
#include "utils/Definitions.h"

namespace tomcat {
    namespace model {

        /**
         * A categorical CPD consists of a table containing the probabilities
         * p(column | row). The number of rows is given by the product of the
         * cardinalities of the parent nodes of the node that is sampled from
         * this CPD. Each row represents a combination of possible assignments
         * of the parent nodes ordered in ascending order with respect to the
         * binary basis. A categorical CPD is used for discrete probabilities.
         * The table is represented by a list of categorical distributions that
         * contain a list of probabilities in itself.
         *
         * For instance,
         *
         * Let A and B be parents of the node C.
         *
         * A -> C B -> C
         *
         * Suppose A, B and C have cardinalities 2, 3 and 4 respectively.
         *
         * Let p(C = c | A = a, B = b) be p(c|a,b). A CategoricalCPD for C will
         * be as follows,
         * _____________________________________________________
         * |///| C |     0    |     1    |     2    |     3    |
         * |---------------------------------------------------|
         * | A | B |///////////////////////////////////////////|
         * |---------------------------------------------------|
         * | 0 | 0 | p(0|0,0) | p(1|0,0) | p(2|0,0) | p(3|0,0) |
         * |---------------------------------------------------|
         * | 0 | 1 | p(0|0,1) | p(1|0,1) | p(2|0,1) | p(3|0,1) |
         * |---------------------------------------------------|
         * | 0 | 2 | p(0|0,2) | p(1|0,2) | p(2|0,2) | p(3|0,2) |
         * |---------------------------------------------------|
         * | 1 | 0 | p(0|1,0) | p(1|1,0) | p(2|1,0) | p(3|1,0) |
         * |---------------------------------------------------|
         * | 1 | 1 | p(0|1,1) | p(1|1,1) | p(2|1,1) | p(3|1,1) |
         * |---------------------------------------------------|
         * | 1 | 2 | p(0|1,2) | p(1|1,2) | p(2|1,2) | p(3|1,2) |
         * |---------------------------------------------------|
         */
        class CategoricalCPD : public CPD {
          public:
            //------------------------------------------------------------------
            // Constructors & Destructor
            //------------------------------------------------------------------

            /**
             * Creates an instance of a Categorical CPD.
             *
             * @param parent_node_order: evaluation order of the parent
             * nodes' assignments for correct distribution indexing
             * @param distributions: list of categorical distributions
             */
            CategoricalCPD(
                const std::vector<std::shared_ptr<NodeMetadata>>&
                    parent_node_order,
                const std::vector<std::shared_ptr<Categorical>>& distributions);

            /**
             * Creates an instance of a Categorical CPD.
             *
             * @param parent_node_order: evaluation order of the parent
             * nodes' assignments for correct distribution indexing
             * @param distributions: list of categorical distributions
             */
            CategoricalCPD(
                std::vector<std::shared_ptr<NodeMetadata>>&& parent_node_order,
                const std::vector<std::shared_ptr<Categorical>>& distributions);

            /**
             * Creates an instance of a Categorical CPD by transforming a
             * table of probabilities to a list of categorical distributions
             * with constant probabilities.
             *
             * @param parent_node_order: evaluation order of the parent
             * nodes' assignments for correct distribution indexing
             * @param probabilities: matrix containing probabilities
             */
            CategoricalCPD(const std::vector<std::shared_ptr<NodeMetadata>>&
                               parent_node_order,
                           const Eigen::MatrixXd& probabilities);

            /**
             * Creates an instance of a Categorical CPD by transforming a
             * table of probabilities to a list of categorical distributions
             * with constant probabilities.
             *
             * @param parent_node_order: evaluation order of the parent
             * nodes' assignments for correct distribution indexing
             * @param probabilities: matrix containing probabilities
             */
            CategoricalCPD(
                std::vector<std::shared_ptr<NodeMetadata>>&& parent_node_order,
                const Eigen::MatrixXd& probabilities);

            ~CategoricalCPD();

            //------------------------------------------------------------------
            // Copy & Move constructors/assignments
            //------------------------------------------------------------------
            CategoricalCPD(const CategoricalCPD& cpd);

            CategoricalCPD& operator=(const CategoricalCPD& cpd);

            CategoricalCPD(CategoricalCPD&& cpd) = default;

            CategoricalCPD& operator=(CategoricalCPD&& cpd) = default;

            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------
            std::unique_ptr<CPD> clone() const override;

            std::string get_name() const override;

            void add_to_sufficient_statistics(
                const std::vector<double>& values) override;

            Eigen::MatrixXd sample_from_conjugacy(
                const std::shared_ptr<gsl_rng>& random_generator,
                int num_samples,
                const std::shared_ptr<const RandomVariableNode>& cpd_owner)
                const override;

            void reset_sufficient_statistics() override;

            Eigen::MatrixXd get_posterior_weights(
                const std::vector<std::shared_ptr<Node>>& index_nodes,
                const std::shared_ptr<RandomVariableNode>& sampled_node,
                const std::shared_ptr<const RandomVariableNode>& cpd_owner,
                int num_jobs) const override;

            std::shared_ptr<CPD>
            create_from_data(const EvidenceSet& data,
                             const std::string& cpd_owner_label,
                             int cpd_owner_cardinality) override;

            bool is_continuous() const override;

            Eigen::VectorXd
            get_pdfs(const std::shared_ptr<const RandomVariableNode>& cpd_owner,
                     int num_jobs,
                     int parameter_idx) const override;

          protected:
            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------
            void clone_distributions() override;

          private:
            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            /**
             * Initialized the CPD from a list of distributions.
             *
             * @param distributions: list of categorical distributions.
             */
            void init_from_distributions(
                const std::vector<std::shared_ptr<Categorical>>& distributions);

            /**
             * Uses the values in the matrix to create a list of constant
             * categorical distributions.
             *
             * @param matrix: matrix of probabilities
             */
            void init_from_matrix(const Eigen::MatrixXd& matrix);

            /**
             * Computes the posterior weights for a given node that owns this
             * CPD.
             *
             * @param cpd_owner: node that owns the CPD
             * @param distribution_indices: indices of the distributions
             * indexed by the parents of the cpd owner
             * @param cardinality: cardinality of the node to which posterior
             * weights are being computed
             * @param distribution_index_offset: how many indices need to be
             * skipped to reach the next sampled node possible value (the
             * multiplicative cardinality of the indexing nodes to the right
             * of the sampled node)
             * @param distributions_table: table os probabilities that
             * represent this categorical CPD
             * @param num_jobs: number of jobs used to compute the weights
             * (if > 1, the computation is performed in multiple threads)
             *
             * @return Posterior weights.
             */
            Eigen::MatrixXd compute_posterior_weights(
                const std::shared_ptr<const RandomVariableNode>& cpd_owner,
                const Eigen::VectorXi& distribution_indices,
                int cardinality,
                int distribution_index_offset,
                const Eigen::MatrixXd& distributions_table,
                int num_jobs) const;

            /**
             * Computes a portion of the posterior weights for a given node.
             *
             * @param cpd_owner: node that owns the CPD
             * @param distribution_indices: indices of the distributions
             * indexed by the parents of the cpd owner
             * @param distribution_index_offset: how many indices need to be
             * skipped to reach the next sampled node possible value (the
             * multiplicative cardinality of the indexing nodes to the right
             * of the sampled node)
             * @param distributions_table: table os probabilities that
             * represent this categorical CPD
             * @param processing_block: initial row and number of rows from
             * the node's assignment to consider for computation
             * @param full_weights: matrix containing the full-weights. A
             * portion of it will be updated by this method
             * @param weights_mutex: mutex to lock the full_weights matrix
             * when this method writes to it
             */
            void run_posterior_weights_thread(
                const std::shared_ptr<const RandomVariableNode>& cpd_owner,
                const Eigen::VectorXi& distribution_indices,
                int distribution_index_offset,
                const Eigen::MatrixXd& distributions_table,
                const std::pair<int, int>& processing_block,
                Eigen::MatrixXd& full_weights,
                std::mutex& weights_mutex) const;

            /**
             * Computes pdfs for the assignments of a CPD owner in a separate
             * thread.
             *
             * @param cpd_owner: node that owns the CPD
             * @param distribution_indices: indices of the distributions from
             * this CPD to be used
             * @param parameter_idx: row of the node's assignment that holds the
             * parameters of the distribution
             * @param full_pdfs: vector of pdfs to be updated by this function
             * @param processing_block: block of data to process in this thread
             * @param pdf_mutex: mutex to control writing in the
             * full_pdfs vector
             */
            void run_pdf_thread(
                const std::shared_ptr<const RandomVariableNode>& cpd_owner,
                const Eigen::VectorXi& distribution_indices,
                const Eigen::MatrixXd& cpd_table,
                int parameter_idx,
                Eigen::VectorXd& full_pdfs,
                const std::pair<int, int>& processing_block,
                std::mutex& pdf_mutex) const;
        };

    } // namespace model
} // namespace tomcat
