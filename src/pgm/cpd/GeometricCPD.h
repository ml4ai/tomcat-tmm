#pragma once

#include "distribution/Geometric.h"
#include "pgm/RandomVariableNode.h"
#include "pgm/cpd/CPD.h"
#include "utils/Definitions.h"

namespace tomcat {
    namespace model {

        /**
         * A Geometric CPD consists of a table containing a list of Geometric
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
         * from a Geometric distribution with parameter \f$p\f$.
         *
         * A CPD for C will be as follows,
         * _________________________________________________
         * |///|                     C                     |
         * |-----------------------------------------------|
         * | A | B |///////////////////////////////////////|
         * |-----------------------------------------------|
         * | 0 | 0 |     Geometric(\f$p_{00}\f$))          |
         * |-----------------------------------------------|
         * | 0 | 1 |     Geometric(\f$p_{01}\f$)           |
         * |-----------------------------------------------|
         * | 0 | 2 |     Geometric(\f$p_{02}\f$)           |
         * |-----------------------------------------------|
         * | 1 | 0 |     Geometric(\f$p_{10}\f$)           |
         * |-----------------------------------------------|
         * | 1 | 1 |     Geometric(\f$p_{11}\f$)           |
         * |-----------------------------------------------|
         * | 1 | 2 |     Geometric(\f$p_{12}\f$)           |
         * |-----------------------------------------------|
         */
        class GeometricCPD : public CPD {
          public:
            //------------------------------------------------------------------
            // Constructors & Destructor
            //------------------------------------------------------------------

            /**
             * Creates an instance of a Geometric CPD.
             *
             * @param parent_node_order: evaluation order of the parent
             * nodes' assignments for correct distribution indexing
             * @param distributions: list of geometric distributions
             */
            GeometricCPD(
                const std::vector<std::shared_ptr<NodeMetadata>>&
                    parent_node_order,
                const std::vector<std::shared_ptr<Geometric>>& distributions);

            /**
             * Creates an instance of a Geometric CPD.
             *
             * @param parent_node_order: evaluation order of the parent
             * nodes' assignments for correct distribution indexing
             * @param distributions: list of geometric distributions
             */
            GeometricCPD(
                std::vector<std::shared_ptr<NodeMetadata>>&& parent_node_order,
                const std::vector<std::shared_ptr<Geometric>>& distributions);

            /**
             * Creates an instance of a Geometric CPD by transforming a
             * vector of ps into a list of Geometric distributions each
             * with one of the elements in the parameter vector.
             *
             * @param parent_node_order: evaluation order of the parent
             * nodes' assignments for correct distribution indexing
             * @param ps: vector containing the parameters p of the
             * Geometric distributions
             */
            GeometricCPD(const std::vector<std::shared_ptr<NodeMetadata>>&
                           parent_node_order,
                       const Eigen::VectorXd& ps);

            /**
             * Creates an instance of a Geometric CPD by transforming a
             * vector of ps into a list of Geometric distributions each
             * with one of the elements in the parameter vector.
             *
             * @param parent_node_order: evaluation order of the parent
             * nodes' assignments for correct distribution indexing
             * @param ps: vector containing the parameters p of the
             * Geometric distributions
             */
            GeometricCPD(
                std::vector<std::shared_ptr<NodeMetadata>>&& parent_node_order,
                const Eigen::VectorXd& probabilities);

            ~GeometricCPD();

            //------------------------------------------------------------------
            // Copy & Move constructors/assignments
            //------------------------------------------------------------------
            GeometricCPD(const GeometricCPD& cpd);

            GeometricCPD& operator=(const GeometricCPD& cpd);

            GeometricCPD(GeometricCPD&& cpd) = default;

            GeometricCPD& operator=(GeometricCPD&& cpd) = default;

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
             * @param distributions: list of Geometric distributions.
             */
            void init_from_distributions(
                const std::vector<std::shared_ptr<Geometric>>& Geometric);

            /**
             * Uses the values in the parameter vector to create a list of
             * constant Geometric distributions.
             *
             * @param ps: parameter vector
             */
            void init_from_vector(const Eigen::VectorXd& ps);
        };

    } // namespace model
} // namespace tomcat
