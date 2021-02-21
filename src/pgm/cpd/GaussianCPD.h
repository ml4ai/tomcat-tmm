#pragma once

#include "distribution/Gaussian.h"
#include "pgm/cpd/CPD.h"
#include "utils/Definitions.h"

namespace tomcat {
    namespace model {

        /**
         * A Gaussian CPD consists of a table containing a list of Gaussian
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
         * from a Gaussian distribution with parameters mean and var.
         *
         * A CPD for C will be as follows,
         * ______________________________________________________
         * |///|                       C                        |
         * |----------------------------------------------------|
         * | A | B |////////////////////////////////////////////|
         * |----------------------------------------------------|
         * | 0 | 0 |     Gaussian(\f$mean_{00}, var_{00}\f$)    |
         * |----------------------------------------------------|
         * | 0 | 1 |     Gaussian(\f$mean_{01}, var_{01}\f$)    |
         * |----------------------------------------------------|
         * | 0 | 2 |     Gaussian(\f$mean_{02}, var_{02}\f$)    |
         * |----------------------------------------------------|
         * | 1 | 0 |     Gaussian(\f$mean_{10}, var_{10}\f$)    |
         * |----------------------------------------------------|
         * | 1 | 1 |     Gaussian(\f$mean_{11}, var_{11}\f$)    |
         * |----------------------------------------------------|
         * | 1 | 2 |     Gaussian(\f$mean_{12}, var_{12}\f$)    |
         * |----------------------------------------------------|
         */
        class GaussianCPD : public CPD {
          public:
            //------------------------------------------------------------------
            // Constructors & Destructor
            //------------------------------------------------------------------

            /**
             * Creates an instance of a Gaussian CPD.
             *
             * @param parent_node_order: evaluation order of the parent
             * nodes' assignments for correct distribution indexing
             * @param distributions: list of Gaussian distributions
             */
            GaussianCPD(
                const std::vector<std::shared_ptr<NodeMetadata>>&
                    parent_node_order,
                const std::vector<std::shared_ptr<Gaussian>>& distributions);

            /**
             * Creates an instance of a Gaussian CPD.
             *
             * @param parent_node_order: evaluation order of the parent
             * nodes' assignments for correct distribution indexing
             * @param distributions: list of Gaussian distributions
             */
            GaussianCPD(
                std::vector<std::shared_ptr<NodeMetadata>>&& parent_node_order,
                std::vector<std::shared_ptr<Gaussian>>&& distributions);

            /**
             * Creates an instance of a Gaussian CPD table by transforming a
             * table of parameter values to a list of Gaussian distributions
             * with constant mean and variance.
             *
             * @param parent_node_order: evaluation order of the parent
             * nodes' assignments for correct distribution indexing
             * @param cpd_table: matrix containing the means and variances
             */
            GaussianCPD(const std::vector<std::shared_ptr<NodeMetadata>>&
                            parent_node_order,
                        const Eigen::MatrixXd& parameters);

            /**
             * Creates an instance of a Gaussian CPD table by transforming a
             * table of parameter values to a list of Gaussian distributions
             * with constant mean and variance.
             *
             * @param parent_node_order: evaluation order of the parent
             * nodes' assignments for correct distribution indexing
             * @param cpd_table: matrix containing the means and variances
             */
            GaussianCPD(
                std::vector<std::shared_ptr<NodeMetadata>>&& parent_node_order,
                const Eigen::MatrixXd& parameters);

            ~GaussianCPD();

            //------------------------------------------------------------------
            // Copy & Move constructors/assignments
            //------------------------------------------------------------------
            GaussianCPD(const GaussianCPD& cpd);

            GaussianCPD& operator=(const GaussianCPD& cpd);

            GaussianCPD(GaussianCPD&& cpd) = default;

            GaussianCPD& operator=(GaussianCPD&& cpd) = default;

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

            bool is_continuous() const override;

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
             * Gaussian distributions.
             *
             * @param matrix: matrix of \f$\alpha\f$s
             */
            virtual void init_from_matrix(const Eigen::MatrixXd& matrix);
        };

    } // namespace model
} // namespace tomcat
