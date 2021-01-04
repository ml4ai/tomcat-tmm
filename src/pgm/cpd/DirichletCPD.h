#pragma once

#include "distribution/Dirichlet.h"
#include "pgm/cpd/CPD.h"
#include "utils/Definitions.h"

namespace tomcat {
    namespace model {

        /**
         * A Dirichlet CPD consists of a table containing a list of Dirichlet
         * distributions. The number of rows is given by the product of the
         * cardinalities of the parent nodes of the node that owns this CPD.
         * Each row represents a combination of possible assignments of these
         * parent nodes.
         *
         * For instance,
         *
         * Let A and B be parents of the node C. Let A and B be discrete
         * values sampled from a finite discrete distribution.
         *
         * Let A, B and C, have the following dependencies:
         * A -> C, B -> C
         *
         * Suppose A, B have cardinalities 2, 3 respectively and C is sampled
         * from a Dirichlet distribution with parameter \f$\alpha\f$.
         *
         * A CPD for C will be as follows,
         * _________________________________________________
         * |///|                      C                    |
         * |-----------------------------------------------|
         * | A | B |///////////////////////////////////////|
         * |-----------------------------------------------|
         * | 0 | 0 |     Dirichlet(\f$\alpha_{00}\f$))     |
         * |-----------------------------------------------|
         * | 0 | 1 |     Dirichlet(\f$\alpha_{01}\f$)      |
         * |-----------------------------------------------|
         * | 0 | 2 |     Dirichlet(\f$\alpha_{02}\f$)      |
         * |-----------------------------------------------|
         * | 1 | 0 |     Dirichlet(\f$\alpha_{10}\f$)      |
         * |-----------------------------------------------|
         * | 1 | 1 |     Dirichlet(\f$\alpha_{11}\f$)      |
         * |-----------------------------------------------|
         * | 1 | 2 |     Dirichlet(\f$\alpha_{12}\f$)      |
         * |-----------------------------------------------|
         */
        class DirichletCPD : public CPD {
          public:
            //------------------------------------------------------------------
            // Constructors & Destructor
            //------------------------------------------------------------------

            /**
             * Creates an instance of a Dirichlet CPD.
             *
             * @param parent_node_order: evaluation order of the parent
             * nodes' assignments for correct distribution indexing
             * @param distributions: list of Dirichlet distributions
             */
            DirichletCPD(
                const std::vector<std::shared_ptr<NodeMetadata>>&
                    parent_node_order,
                const std::vector<std::shared_ptr<Dirichlet>>& distributions);

            /**
             * Creates an instance of a Dirichlet CPD.
             *
             * @param parent_node_order: evaluation order of the parent
             * nodes' assignments for correct distribution indexing
             * @param distributions: list of Dirichlet distributions
             */
            DirichletCPD(
                std::vector<std::shared_ptr<NodeMetadata>>&& parent_node_order,
                const std::vector<std::shared_ptr<Dirichlet>>& distributions);

            /**
             * Creates an instance of a Dirichlet CPD table by transforming a
             * table of parameter values to a list of Dirichlet distributions
             * with constant parameters.
             *
             * @param parent_node_order: evaluation order of the parent
             * nodes' assignments for correct distribution indexing
             * @param cpd_table: matrix containing a \f$\alpha\f$s
             */
            DirichletCPD(const std::vector<std::shared_ptr<NodeMetadata>>&
                             parent_node_order,
                         const Eigen::MatrixXd& alphas);

            /**
             * Creates an instance of a Dirichlet CPD table by transforming a
             * table of parameter values to a list of Dirichlet distributions
             * with constant parameters.
             *
             * @param parent_node_order: evaluation order of the parent
             * nodes' assignments for correct distribution indexing
             * @param cpd_table: matrix containing a \f$\alpha\f$s
             */
            DirichletCPD(
                std::vector<std::shared_ptr<NodeMetadata>>&& parent_node_order,
                const Eigen::MatrixXd& alphas);

            ~DirichletCPD();

            //------------------------------------------------------------------
            // Copy & Move constructors/assignments
            //------------------------------------------------------------------
            DirichletCPD(const DirichletCPD& cpd);

            DirichletCPD& operator=(const DirichletCPD& cpd);

            DirichletCPD(DirichletCPD&& cpd) = default;

            DirichletCPD& operator=(DirichletCPD&& cpd) = default;

            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------
            std::unique_ptr<CPD> clone() const override;

            std::string get_description() const override;

            void add_to_sufficient_statistics(
                const std::vector<double>& values) override;

            Eigen::MatrixXd sample_from_conjugacy(
                const std::shared_ptr<gsl_rng>& random_generator,
                const std::vector<std::shared_ptr<Node>>& parent_nodes,
                int num_samples) const override;

            void reset_sufficient_statistics() override;

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
             * @param distributions: list of Dirichlet distributions.
             */
            void init_from_distributions(
                const std::vector<std::shared_ptr<Dirichlet>>& distributions);

            /**
             * Uses the values in the matrix to create a list of constant
             * Dirichlet distributions.
             *
             * @param matrix: matrix of \f$\alpha\f$s
             */
            void init_from_matrix(const Eigen::MatrixXd& matrix);

            //------------------------------------------------------------------
            // Data members
            //------------------------------------------------------------------

            /**
             * Sufficient statistics used for CPDs owned by parameter nodes.
             * It's used to compute the posterior of a conjugate prior.
             */
            Eigen::VectorXd sufficient_statistics;
        };

    } // namespace model
} // namespace tomcat
