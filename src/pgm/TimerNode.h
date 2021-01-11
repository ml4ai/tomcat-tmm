#pragma once

#include "RandomVariableNode.h"

#include "pgm/cpd/CPD.h"
#include "utils/Definitions.h"

namespace tomcat {
    namespace model {

        //------------------------------------------------------------------
        // Forward declarations
        //------------------------------------------------------------------

        class TimerNode;

        //------------------------------------------------------------------
        // Structs
        //------------------------------------------------------------------

        /**
         * A random timer node is a random variable node that controls the
         * amount of time a dependent node is sampled.
         */
        class TimerNode : public RandomVariableNode {
          public:
            //------------------------------------------------------------------
            // Constructors & Destructor
            //------------------------------------------------------------------

            /**
             * Creates an instance of a timer node.
             *
             * @param metadata: node's metadata
             * @param time_step: node's time step in an unrolled DBN (0 by
             * default)
             */
            TimerNode(const std::shared_ptr<NodeMetadata>& metadata,
                      int time_step = 0);

            /**
             * Creates an instance of a timer node.
             *
             * @param metadata: node's metadata
             * @param time_step: node's time step in an unrolled DBN (0 by
             * default)
             */
            TimerNode(std::shared_ptr<NodeMetadata>&& metadata,
                      int time_step = 0);

            ~TimerNode();

            //------------------------------------------------------------------
            // Copy & Move constructors/assignments
            //------------------------------------------------------------------
            TimerNode(const TimerNode& node);

            TimerNode& operator=(const TimerNode& node);

            TimerNode(TimerNode&&) = default;

            TimerNode& operator=(TimerNode&&) = default;

            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            std::unique_ptr<Node> clone() const override;
            //
            //            std::string get_timed_name() const override;
            //
            //            /**
            //             * Marks the CPDs of the node as not updated.
            //             */
            //            void reset_cpd_updated_status();
            //
            //            /**
            //             * Replaces parameter nodes in node dependent CPD
            //             templates by a
            //             * concrete timed-instance node in the unrolled DBN.
            //             *
            //             * @param parameter_nodes_map: mapping between a
            //             parameter node
            //             * timed name and its object in an unrolled DBN
            //             * @param time_step: time step of the node that owns
            //             the CPD in the
            //             * unrolled DBN. It can be different from the time
            //             step of the
            //             * parameter node if the latter is shared among nodes
            //             over several
            //             * time steps.
            //             */
            //            void update_cpd_templates_dependencies(
            //                const NodeMap& parameter_nodes_map, int
            //                time_step);
            //
            //            /**
            //             * Create new references for the CPD templates of the
            //             node.
            //             */
            //            void clone_cpd_templates();
            //
            //            /**
            //             * Generates samples from this node's CPD given its
            //             parents'
            //             * assignments.
            //             *
            //             * @param random_generator: random number generator
            //             * @param num_samples: number of samples to generate
            //             *
            //             * @return Samples from the node's CPD.
            //             */
            //            Eigen::MatrixXd
            //            sample(const std::shared_ptr<gsl_rng>&
            //            random_generator,
            //                   int num_samples) const;
            //
            //            /**
            //             * Generates a sample for the node from its posterior
            //             distribution.
            //             * If the node is an in-plate node, multiple samples
            //             will be
            //             * generated for the node (one per row). The number of
            //             total
            //             * samples are defined by the number of elements
            //             in-plate.
            //             *
            //             * Note: This method changes this node assignment
            //             temporarily while
            //             * calculating the weights, therefore it's not const.
            //             The final
            //             * state of the this object is unchanged though.
            //             *
            //             * @param random_generator: random number generator
            //             *
            //             * @return Sample for the node from its posterior
            //             */
            //            Eigen::MatrixXd sample_from_posterior(
            //                const std::shared_ptr<gsl_rng>& random_generator);
            //
            //            /**
            //             * Returns p(children(node)|node). The posterior of a
            //             node is
            //             * given by p(node|parents(node)) *
            //             p(children(node)|node). We
            //             * call the second term, the posterior weights here.
            //             *
            //             * Note: This method changes this node assignment
            //             temporarily while
            //             * calculating the weights, therefore it's not const.
            //             The final
            //             * state of the this object is unchanged though.
            //             *
            //             * @return Posterior weights
            //             */
            //            Eigen::MatrixXd get_posterior_weights();
            //
            //            /**
            //             * Samples a node using conjugacy properties and
            //             sufficient
            //             * statistics stored in the node's CPD.
            //             *
            //             * @param random_generator: random number generator
            //             * @return
            //             */
            //            Eigen::MatrixXd sample_from_conjugacy(
            //                const std::shared_ptr<gsl_rng>& random_generator,
            //                const std::vector<std::shared_ptr<Node>>&
            //                parent_nodes, int num_samples) const;
            //
            //            /**
            //             * Update sufficient statistics of parent parameter
            //             nodes with this
            //             * node's assignment(s).
            //             *
            //             */
            //            void update_parents_sufficient_statistics();
            //
            //            /**
            //             * Adds a set of values to the sufficient statistics
            //             of a
            //             * parameter node's CPD.
            //             *
            //             * @param values: Values to add to the sufficient
            //             statistics. The
            //             * update_parents_sufficient_statistics will call this
            //             function for
            //             * parameter nodes at some point.
            //             */
            //            void
            //            add_to_sufficient_statistics(const
            //            std::vector<double>& values);
            //
            //            /**
            //             * Clears the values stored as sufficient statistics
            //             in the
            //             * parameter node CPD.
            //             */
            //            void reset_sufficient_statistics();
            //
            //            /**
            //             * Prevents node's assignment to be changed.
            //             */
            //            void freeze();
            //
            //            /**
            //             * Frees node's assignment to be changed.
            //             */
            //            void unfreeze();
            //
            //            /**
            //             * Adds a CPD to the list of possible CPDs of the
            //             node.
            //             *
            //             * @param cpd: CPD
            //             */
            //            void add_cpd_template(const std::shared_ptr<CPD>&
            //            cpd);
            //
            //            /**
            //             * Adds CPD to the list of possible CPDs of the node.
            //             *
            //             * @param cpd: CPD
            //             */
            //            void add_cpd_template(std::shared_ptr<CPD>&& cpd);
            //
            //            /**
            //             * Returns the node's CPD associated to a set of
            //             parents.
            //             *
            //             * @param parent_labels: labels of the parents of the
            //             node
            //             *
            //             * @return node's CPD related to the parents informed
            //             */
            //            std::shared_ptr<CPD>
            //            get_cpd_for(const std::vector<std::string>&
            //            parent_labels) const;

            // -----------------------------------------------------------------
            // Getters & Setters
            // -----------------------------------------------------------------
            const Eigen::MatrixXd& get_backward_assignment() const;
            void
            set_backward_assignment(const Eigen::MatrixXd& backward_assignment);

          private:
            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            /**
             * Copies data members of a timer node.
             *
             * @param cpd: continuous CPD
             */
            void copy_node(const TimerNode& node);

            // -----------------------------------------------------------------
            // Data members
            // -----------------------------------------------------------------

            // Size of the segments to the right (in time) of the timer node.
            // Assignments of the node this timer node controls are counted
            // backwards in time ans stored in this attribute.
            Eigen::MatrixXd backward_assignment;
        };

    } // namespace model
} // namespace tomcat
