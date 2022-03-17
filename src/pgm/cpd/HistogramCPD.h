#pragma once

#include "distribution/Histogram.h"
#include "pgm/cpd/CPD.h"
#include "utils/Definitions.h"

namespace tomcat {
    namespace model {

        /**
         * An Histogram CPD consists of a table containing a list of Histogram
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
         * | 0 | 0 |     Histogram(\f$samples_{00}\f$)          |
         * |----------------------------------------------------|
         * | 0 | 1 |     Histogram(\f$samples_{01}\f$)          |
         * |----------------------------------------------------|
         * | 0 | 2 |     Histogram(\f$samples_{02}\f$)          |
         * |----------------------------------------------------|
         * | 1 | 0 |     Histogram(\f$samples_{10}\f$)          |
         * |----------------------------------------------------|
         * | 1 | 1 |     Histogram(\f$samples_{11}\f$)          |
         * |----------------------------------------------------|
         * | 1 | 2 |     Histogram(\f$samples_{12}\f$)          |
         * |----------------------------------------------------|
         */
        class HistogramCPD : public CPD {
          public:
            //------------------------------------------------------------------
            // Constructors & Destructor
            //------------------------------------------------------------------

            /**
             * Creates an instance of an histogram CPD.
             *
             * @param parent_node_order: evaluation order of the parent
             * nodes' assignments for correct distribution indexing
             * @param distributions: list of histogram distributions
             */
            HistogramCPD(
                const std::vector<std::shared_ptr<NodeMetadata>>&
                    parent_node_order,
                const std::vector<std::shared_ptr<Histogram>>& distributions);

            /**
             * Creates an instance of an histogram CPD.
             *
             * @param parent_node_order: evaluation order of the parent
             * nodes' assignments for correct distribution indexing
             * @param distributions: list of histogram distributions
             */
            HistogramCPD(
                std::vector<std::shared_ptr<NodeMetadata>>&& parent_node_order,
                std::vector<std::shared_ptr<Histogram>>&& distributions);

            /**
             * Creates an instance of an histogram CPD table by transforming a
             * table containing lists of samples to a list of histogram
             * distributions.
             *
             * @param parent_node_order: evaluation order of the parent
             * nodes' assignments for correct distribution indexing
             * @param samples_table: table containing samples from any
             * distributions
             */
            HistogramCPD(const std::vector<std::shared_ptr<NodeMetadata>>&
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
            HistogramCPD(
                std::vector<std::shared_ptr<NodeMetadata>>&& parent_node_order,
                const std::vector<Eigen::VectorXd>& samples_table);

            ~HistogramCPD();

            //------------------------------------------------------------------
            // Copy & Move constructors/assignments
            //------------------------------------------------------------------
            HistogramCPD(const HistogramCPD& cpd);

            HistogramCPD& operator=(const HistogramCPD& cpd);

            HistogramCPD(HistogramCPD&& cpd) = default;

            HistogramCPD& operator=(HistogramCPD&& cpd) = default;

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
