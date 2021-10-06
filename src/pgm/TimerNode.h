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
             *
             * @return Timer values given controlled nodes' assignment.
             */
            Eigen::MatrixXd sample_from_posterior(
                const std::vector<std::shared_ptr<gsl_rng>>&
                    random_generator_per_job,
                const std::vector<int>& time_steps_per_sample = {}) override;

            /**
             * Gets posterior weights for the left segment of a node such
             * that this timer node is the timer at the beginning of that
             * segment.
             *
             * @param left_segment_duration of the left segment for the
             * sample_idx timer's assignment
             * @param right_segment_state: first state of the right segment
             * @param central_segment_time_step: time step at the central
             * segment
             * @param last_time_step: time step of the last timer being
             * sampled in the unrolled DBN
             * @param sample_idx: row of the assignment to consider when
             * calculating the weights.
             *
             * @return Left segment posterior weights
             */
            Eigen::VectorXd get_left_segment_posterior_weights(
                int left_segment_duration,
                const std::shared_ptr<RandomVariableNode>& right_segment_state,
                int central_segment_time_step,
                int last_time_step,
                int sample_idx) const;

            /**
             * Computes the duration of the segments backwards.
             */
            void update_backward_assignment();

            /**
             * Saves the node's current forward assignment for future usage.
             */
            void stack_forward_assignment();

            /**
             * Sets as the node's forward assignment a previously stacked
             * forward assignment. If there's no assignment stacked, the node's
             * assignment is preserved.
             */
            void pop_forward_assignment();

            // -----------------------------------------------------------------
            // Getters & Setters
            // -----------------------------------------------------------------
            const Eigen::MatrixXd& get_forward_assignment() const;

            Eigen::MatrixXd get_forward_assignment(
                const ProcessingBlock& processing_block) const;

            double get_forward_assignment(int i, int j) const;

            void
            set_forward_assignment(const Eigen::MatrixXd& forward_assignment);

            void
            set_forward_assignment(const Eigen::MatrixXd& forward_assignment,
                                   const ProcessingBlock& processing_block);

            const Eigen::MatrixXd& get_backward_assignment() const;

            Eigen::MatrixXd get_backward_assignment(
                const ProcessingBlock& processing_block) const;

            double get_backward_assignment(int i, int j) const;

            void
            set_backward_assignment(const Eigen::MatrixXd& backward_assignment);

            void
            set_backward_assignment(const Eigen::MatrixXd& backward_assignment,
                                    const ProcessingBlock& processing_block);

            const std::shared_ptr<RandomVariableNode>&
            get_controlled_node() const;

            void set_controlled_node(
                const std::shared_ptr<RandomVariableNode>& controlled_node);

            const Eigen::MatrixXd& get_stacked_forward_assignment() const;

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

            Eigen::MatrixXd stacked_forward_assignment;
        };

    } // namespace model
} // namespace tomcat
