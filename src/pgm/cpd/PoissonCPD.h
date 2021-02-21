#pragma once

#include "distribution/Poisson.h"
#include "pgm/RandomVariableNode.h"
#include "pgm/cpd/CPD.h"
#include "utils/Definitions.h"

namespace tomcat {
    namespace model {

        /**
         * A Poisson CPD consists of a table containing a list of Poisson
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
         * from a Poisson distribution with parameter \f$\lambda\f$.
         *
         * A CPD for C will be as follows,
         * _________________________________________________
         * |///|                     C                     |
         * |-----------------------------------------------|
         * | A | B |///////////////////////////////////////|
         * |-----------------------------------------------|
         * | 0 | 0 |     Poisson(\f$\lambda_{00}\f$))      |
         * |-----------------------------------------------|
         * | 0 | 1 |     Poisson(\f$\lambda_{01}\f$)       |
         * |-----------------------------------------------|
         * | 0 | 2 |     Poisson(\f$\lambda_{02}\f$)       |
         * |-----------------------------------------------|
         * | 1 | 0 |     Poisson(\f$\lambda_{10}\f$)       |
         * |-----------------------------------------------|
         * | 1 | 1 |     Poisson(\f$\lambda_{11}\f$)       |
         * |-----------------------------------------------|
         * | 1 | 2 |     Poisson(\f$\lambda_{12}\f$)       |
         * |-----------------------------------------------|
         */
        class PoissonCPD : public CPD {
          public:
            //------------------------------------------------------------------
            // Constructors & Destructor
            //------------------------------------------------------------------

            /**
             * Creates an instance of a Poisson CPD.
             *
             * @param parent_node_order: evaluation order of the parent
             * nodes' assignments for correct distribution indexing
             * @param distributions: list of categorical distributions
             */
            PoissonCPD(
                const std::vector<std::shared_ptr<NodeMetadata>>&
                    parent_node_order,
                const std::vector<std::shared_ptr<Poisson>>& distributions);

            /**
             * Creates an instance of a Poisson CPD.
             *
             * @param parent_node_order: evaluation order of the parent
             * nodes' assignments for correct distribution indexing
             * @param distributions: list of categorical distributions
             */
            PoissonCPD(
                std::vector<std::shared_ptr<NodeMetadata>>&& parent_node_order,
                const std::vector<std::shared_ptr<Poisson>>& distributions);

            /**
             * Creates an instance of a Poisson CPD by transforming a
             * vector of lambdas into a list of Poisson distributions each
             * with one of the elements in the parameter vector.
             *
             * @param parent_node_order: evaluation order of the parent
             * nodes' assignments for correct distribution indexing
             * @param lambdas: vector containing the parameters lambda of the
             * Poisson distributions
             */
            PoissonCPD(const std::vector<std::shared_ptr<NodeMetadata>>&
                           parent_node_order,
                       const Eigen::VectorXd& lambdas);

            /**
             * Creates an instance of a Poisson CPD by transforming a
             * vector of lambdas into a list of Poisson distributions each
             * with one of the elements in the parameter vector.
             *
             * @param parent_node_order: evaluation order of the parent
             * nodes' assignments for correct distribution indexing
             * @param lambdas: vector containing the parameters lambda of the
             * Poisson distributions
             */
            PoissonCPD(
                std::vector<std::shared_ptr<NodeMetadata>>&& parent_node_order,
                const Eigen::VectorXd& probabilities);

            ~PoissonCPD();

            //------------------------------------------------------------------
            // Copy & Move constructors/assignments
            //------------------------------------------------------------------
            PoissonCPD(const PoissonCPD& cpd);

            PoissonCPD& operator=(const PoissonCPD& cpd);

            PoissonCPD(PoissonCPD&& cpd) = default;

            PoissonCPD& operator=(PoissonCPD&& cpd) = default;

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
             * Initialized the CPD from a list of distributions.
             *
             * @param distributions: list of Poisson distributions.
             */
            void init_from_distributions(
                const std::vector<std::shared_ptr<Poisson>>& poisson);

            /**
             * Uses the values in the parameter vector to create a list of
             * constant Poisson distributions.
             *
             * @param lambdas: parameter vector
             */
            void init_from_vector(const Eigen::VectorXd& lambdas);
        };

    } // namespace model
} // namespace tomcat
