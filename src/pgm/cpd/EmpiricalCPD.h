#pragma once

#include "distribution/Empirical.h"
#include "pgm/cpd/CPD.h"
#include "utils/Definitions.h"

namespace tomcat {
    namespace model {

        /**
         * An Empirical CPD consists of a table containing a list of Empirical
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
         * | 0 | 0 |     Empirical(\f$samples_{00}\f$)          |
         * |----------------------------------------------------|
         * | 0 | 1 |     Empirical(\f$samples_{01}\f$)          |
         * |----------------------------------------------------|
         * | 0 | 2 |     Empirical(\f$samples_{02}\f$)          |
         * |----------------------------------------------------|
         * | 1 | 0 |     Empirical(\f$samples_{10}\f$)          |
         * |----------------------------------------------------|
         * | 1 | 1 |     Empirical(\f$samples_{11}\f$)          |
         * |----------------------------------------------------|
         * | 1 | 2 |     Empirical(\f$samples_{12}\f$)          |
         * |----------------------------------------------------|
         */
        class EmpiricalCPD : public CPD {
          public:
            //------------------------------------------------------------------
            // Constructors & Destructor
            //------------------------------------------------------------------

            /**
             * Creates an instance of an empirical CPD.
             *
             * @param parent_node_order: evaluation order of the parent
             * nodes' assignments for correct distribution indexing
             * @param distributions: list of empirical distributions
             */
            EmpiricalCPD(
                const std::vector<std::shared_ptr<NodeMetadata>>&
                    parent_node_order,
                const std::vector<std::shared_ptr<Empirical>>& distributions);

            /**
             * Creates an instance of an empirical CPD.
             *
             * @param parent_node_order: evaluation order of the parent
             * nodes' assignments for correct distribution indexing
             * @param distributions: list of empirical distributions
             */
            EmpiricalCPD(
                std::vector<std::shared_ptr<NodeMetadata>>&& parent_node_order,
                std::vector<std::shared_ptr<Empirical>>&& distributions);

            /**
             * Creates an instance of an empirical CPD table by transforming a
             * table containing lists of samples to a list of empirical
             * distributions.
             *
             * @param parent_node_order: evaluation order of the parent
             * nodes' assignments for correct distribution indexing
             * @param samples_table: table containing samples from any
             * distributions
             */
            EmpiricalCPD(const std::vector<std::shared_ptr<NodeMetadata>>&
                             parent_node_order,
                         const std::vector<Eigen::VectorXd>& samples_table);

            /**
             * Creates an instance of a Gaussian CPD table by transforming a
             * table of parameter values to a list of Gaussian distributions
             * with constant mean and variance.
             *
             * @param parent_node_order: evaluation order of the parent
             * nodes' assignments for correct distribution indexing
             * @param samples_table: table containing samples from any
             * distributions
             */
            EmpiricalCPD(
                std::vector<std::shared_ptr<NodeMetadata>>&& parent_node_order,
                const std::vector<Eigen::VectorXd>& samples_table);

            ~EmpiricalCPD();

            //------------------------------------------------------------------
            // Copy & Move constructors/assignments
            //------------------------------------------------------------------
            EmpiricalCPD(const EmpiricalCPD& cpd);

            EmpiricalCPD& operator=(const EmpiricalCPD& cpd);

            EmpiricalCPD(EmpiricalCPD&& cpd) = default;

            EmpiricalCPD& operator=(EmpiricalCPD&& cpd) = default;

            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------
            std::unique_ptr<CPD> clone() const override;

            std::string get_name() const override;

            void add_to_sufficient_statistics(
                const std::shared_ptr<const Distribution>& distribution,
                const std::vector<double>& values) override;

            Eigen::MatrixXd sample_from_conjugacy(
                const std::shared_ptr<gsl_rng>& random_generator,
                int num_samples,
                const std::shared_ptr<const RandomVariableNode>& cpd_owner)
                const override;

            void reset_sufficient_statistics() override;

            bool is_continuous() const override;

            void update_sufficient_statistics(
                const std::shared_ptr<RandomVariableNode>& cpd_owner) override;

            void freeze_distributions(int parameter_idx) override;

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
             * Uses the values in the samples table to create a list of constant
             * empirical distributions.
             *
             * @param samples_table: table with samples from any distributions
             */
            void
            init_from_table(const std::vector<Eigen::VectorXd>& samples_table);
        };

    } // namespace model
} // namespace tomcat
