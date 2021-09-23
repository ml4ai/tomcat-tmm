#pragma once

#include "distribution/InverseGamma.h"
#include "pgm/cpd/CPD.h"
#include "utils/Definitions.h"

namespace tomcat {
    namespace model {

        /**
         * An InverseGamma CPD consists of a table containing a list of InverseGamma
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
         * from a InverseGamma distribution with parameter \f$\alpha\f$.
         *
         * A CPD for C will be as follows,
         * _________________________________________________
         * |///|                      C                    |
         * |-----------------------------------------------|
         * | A | B |///////////////////////////////////////|
         * |-----------------------------------------------|
         * | 0 | 0 |     InverseGamma(a_{00},b_{00})       |
         * |-----------------------------------------------|
         * | 0 | 1 |     InverseGamma(a_{01},b_{01})       |
         * |-----------------------------------------------|
         * | 0 | 2 |     InverseGamma(a_{02},b_{02})       |
         * |-----------------------------------------------|
         * | 1 | 0 |     InverseGamma(a_{10},b_{10})       |
         * |-----------------------------------------------|
         * | 1 | 1 |     InverseGamma(a_{11},b_{11})       |
         * |-----------------------------------------------|
         * | 1 | 2 |     InverseGamma(a_{12},b_{12})       |
         * |-----------------------------------------------|
         */
        class InverseGammaCPD : public CPD {
          public:
            //------------------------------------------------------------------
            // Constructors & Destructor
            //------------------------------------------------------------------

            /**
             * Creates an instance of a InverseGamma CPD.
             *
             * @param parent_node_order: evaluation order of the parent
             * nodes' assignments for correct distribution indexing
             * @param distributions: list of InverseGamma distributions
             */
            InverseGammaCPD(
                const std::vector<std::shared_ptr<NodeMetadata>>&
                    parent_node_order,
                const std::vector<std::shared_ptr<InverseGamma>>& distributions);

            /**
             * Creates an instance of a InverseGamma CPD.
             *
             * @param parent_node_order: evaluation order of the parent
             * nodes' assignments for correct distribution indexing
             * @param distributions: list of InverseGamma distributions
             */
            InverseGammaCPD(
                std::vector<std::shared_ptr<NodeMetadata>>&& parent_node_order,
                const std::vector<std::shared_ptr<InverseGamma>>& distributions);

            /**
             * Creates an instance of a InverseGamma CPD table by transforming a
             * table of parameter values to a list of InverseGamma distributions
             * with constant parameters.
             *
             * @param parent_node_order: evaluation order of the parent
             * nodes' assignments for correct distribution indexing
             * @param cpd_table: matrix containing the parameters a and b
             */
            InverseGammaCPD(const std::vector<std::shared_ptr<NodeMetadata>>&
                             parent_node_order,
                         const Eigen::MatrixXd& parameters);

            /**
             * Creates an instance of a InverseGamma CPD table by transforming a
             * table of parameter values to a list of InverseGamma distributions
             * with constant parameters.
             *
             * @param parent_node_order: evaluation order of the parent
             * nodes' assignments for correct distribution indexing
             * @param cpd_table: matrix containing the parameters a and b
             */
            InverseGammaCPD(
                std::vector<std::shared_ptr<NodeMetadata>>&& parent_node_order,
                const Eigen::MatrixXd& parameters);

            ~InverseGammaCPD();

            //------------------------------------------------------------------
            // Copy & Move constructors/assignments
            //------------------------------------------------------------------
            InverseGammaCPD(const InverseGammaCPD& cpd);

            InverseGammaCPD& operator=(const InverseGammaCPD& cpd);

            InverseGammaCPD(InverseGammaCPD&& cpd) = default;

            InverseGammaCPD& operator=(InverseGammaCPD&& cpd) = default;

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
             * @param distributions: list of InverseGamma distributions.
             */
            void init_from_distributions(
                const std::vector<std::shared_ptr<InverseGamma>>& distributions);

            /**
             * Uses the values in the matrix to create a list of constant
             * InverseGamma distributions.
             *
             * @param matrix: matrix of \f$\alpha\f$s
             */
            void init_from_matrix(const Eigen::MatrixXd& matrix);
        };

    } // namespace model
} // namespace tomcat
