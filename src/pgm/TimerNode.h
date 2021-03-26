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

            /**
             * Increment timer assignment if subsequent controlled nodes are in
             * the same state. Otherwise, restart its value.
             *
             * @param random_generator_per_job: random number generator per
             * thread
             * @param num_jobs: The computations in this method is not
             * performed in parallel as they can be achieved by small matrix
             * operations. Therefore, this parameter is irrelevant in this
             * subclass.
             * @param min_time_step_to_sample: not used in the timer node
             * computation of the posterior
             * @param max_time_step_to_sample: ignore segments with time step
             * larger than this value
             * @param use_weights_cache: not used in the timer node
             * computation of the posterior
             *
             * @return Timer values given controlled nodes' assignment.
             */
            Eigen::MatrixXd
            sample_from_posterior(const std::vector<std::shared_ptr<gsl_rng>>&
                                      random_generator_per_job,
                                  int min_time_step_to_sample,
                                  int max_time_step_to_sample,
                                  bool use_weights_cache) override;

            /**
             * Gets posterior weights for the left segment of a node such
             * that this timer node is the timer at the beginning of that
             * segment.
             *
             * @param left_segment_duration of the left segment for the
             * sample_idx timer's assignment
             * @param sample_idx: row of the assignment to consider when
             * calculating the weights.
             *
             * @return Left segment posterior weights
             */
            Eigen::VectorXd get_left_segment_posterior_weights(
                const std::shared_ptr<const RandomVariableNode>&
                    right_segment_state,
                int left_segment_duration,
                int last_time_step,
                int sample_idx) const;

            /**
             * Computes the duration of the segments backwards.
             *
             * @param max_time_step_to_sample: if this node's time step if
             * equal to the value informed here, consider it the last node of
             * the model
             */
            void update_backward_assignment(int max_time_step_to_sample);

            // -----------------------------------------------------------------
            // Getters & Setters
            // -----------------------------------------------------------------
            const Eigen::MatrixXd& get_forward_assignment() const;

            void
            set_forward_assignment(const Eigen::MatrixXd& forward_assignment);

            const Eigen::MatrixXd& get_backward_assignment() const;

            void
            set_backward_assignment(const Eigen::MatrixXd& backward_assignment);

            const std::shared_ptr<RandomVariableNode>&
            get_controlled_node() const;

            void set_controlled_node(
                const std::shared_ptr<RandomVariableNode>& controlled_node);

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

            // Size of the segments (in time) to the left of the
            // timer node. Assignments of the node this timer node controls are
            // counted forward in time and stored in this attribute.
            Eigen::MatrixXd forward_assignment;

            // Timed copy of the node controlled by this timer.
            std::shared_ptr<RandomVariableNode> controlled_node;
        };

    } // namespace model
} // namespace tomcat
