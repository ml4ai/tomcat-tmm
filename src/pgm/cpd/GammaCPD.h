#pragma once

#include "distribution/Gamma.h"
#include "pgm/cpd/CPD.h"
#include "utils/Definitions.h"

namespace tomcat {
    namespace model {

        /**
         * A Gamma CPD consists of a table containing a list of Gamma
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
         * from a Gamma distribution with parameters alpha and beta.
         *
         * A CPD for C will be as follows,
         * _______________________________________________________
         * |///|                         C                       |
         * |-----------------------------------------------------|
         * | A | B |/////////////////////////////////////////////|
         * |-----------------------------------------------------|
         * | 0 | 0 |     Gamma(\f$\alpha_{00}, \beta_{00}\f$)    |
         * |-----------------------------------------------------|
         * | 0 | 1 |     Gamma(\f$\alpha_{01}, \beta_{01}\f$)    |
         * |-----------------------------------------------------|
         * | 0 | 2 |     Gamma(\f$\alpha_{02}, \beta_{02}\f$)    |
         * |-----------------------------------------------------|
         * | 1 | 0 |     Gamma(\f$\alpha_{10}, \beta_{10}\f$)    |
         * |-----------------------------------------------------|
         * | 1 | 1 |     Gamma(\f$\alpha_{11}, \beta_{11}\f$)    |
         * |-----------------------------------------------------|
         * | 1 | 2 |     Gamma(\f$\alpha_{12}, \beta_{12}\f$)    |
         * |-----------------------------------------------------|
         */
        class GammaCPD : public CPD {
          public:
            //------------------------------------------------------------------
            // Constructors & Destructor
            //------------------------------------------------------------------

            /**
             * Creates an instance of a Gamma CPD.
             *
             * @param parent_node_order: evaluation order of the parent
             * nodes' assignments for correct distribution indexing
             * @param distributions: list of Gamma distributions
             */
            GammaCPD(const std::vector<std::shared_ptr<NodeMetadata>>&
                         parent_node_order,
                     const std::vector<std::shared_ptr<Gamma>>& distributions);

            /**
             * Creates an instance of a Gamma CPD.
             *
             * @param parent_node_order: evaluation order of the parent
             * nodes' assignments for correct distribution indexing
             * @param distributions: list of Gamma distributions
             */
            GammaCPD(
                std::vector<std::shared_ptr<NodeMetadata>>&& parent_node_order,
                std::vector<std::shared_ptr<Gamma>>&& distributions);

            /**
             * Creates an instance of a Gamma CPD table by transforming a
             * table of parameter values to a list of Gamma distributions
             * with constant mean and variance.
             *
             * @param parent_node_order: evaluation order of the parent
             * nodes' assignments for correct distribution indexing
             * @param cpd_table: matrix containing the parameters alpha and
             * beta of the list of distributions
             */
            GammaCPD(const std::vector<std::shared_ptr<NodeMetadata>>&
                         parent_node_order,
                     const Eigen::MatrixXd& parameters);

            /**
             * Creates an instance of a Gamma CPD table by transforming a
             * table of parameter values to a list of Gamma distributions
             * with constant mean and variance.
             *
             * @param parent_node_order: evaluation order of the parent
             * nodes' assignments for correct distribution indexing
             * @param cpd_table: matrix containing the parameters alpha and
             * beta of the list of distributions
             */
            GammaCPD(
                std::vector<std::shared_ptr<NodeMetadata>>&& parent_node_order,
                const Eigen::MatrixXd& parameters);

            ~GammaCPD();

            //------------------------------------------------------------------
            // Copy & Move constructors/assignments
            //------------------------------------------------------------------
            GammaCPD(const GammaCPD& cpd);

            GammaCPD& operator=(const GammaCPD& cpd);

            GammaCPD(GammaCPD&& cpd) = default;

            GammaCPD& operator=(GammaCPD&& cpd) = default;

            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------
            std::unique_ptr<CPD> clone() const override;

            std::string get_description() const override;

            void add_to_sufficient_statistics(
                const std::vector<double>& values) override;

            Eigen::MatrixXd sample_from_conjugacy(
                const std::shared_ptr<gsl_rng>& random_generator,
                int num_samples,
                const std::shared_ptr<const RandomVariableNode>& cpd_owner)
                const override;

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
             * Uses the values in the matrix to create a list of constant
             * Gamma distributions.
             *
             * @param matrix: matrix of \f$\alpha\f$s
             */
            virtual void init_from_matrix(const Eigen::MatrixXd& matrix);

            //------------------------------------------------------------------
            // Data members
            //------------------------------------------------------------------

            /**
             * Sufficient statistics used for CPDs owned by parameter nodes.
             * It's used to compute the posterior of a conjugate prior. For
             * the Gamma distribution, it consists of a 2D vector with the
             * additive values for alpha and beta parameters of the prior.
             */
            Eigen::VectorXd sufficient_statistics;
        };

    } // namespace model
} // namespace tomcat
