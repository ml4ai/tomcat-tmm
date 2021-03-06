#pragma once

#include <utility>

#include "pgm/inference/FactorNode.h"

namespace tomcat {
    namespace model {

        /**
         * Represents a factor node that accounts for expanding previous
         * segments to account for probabilities of different past segment
         * durations.
         */
        class SegmentExpansionFactorNode : public FactorNode {
          public:
            //------------------------------------------------------------------
            // Constructors & Destructor
            //------------------------------------------------------------------

            /**
             * Creates factor node to compute probabilities of closed
             * and extended left segments.
             */
            SegmentExpansionFactorNode(
                const std::string& label,
                int time_step,
                const DistributionPtrVec& duration_distributions,
                const CPD::TableOrderingMap& duration_ordering_map,
                const CPD::TableOrderingMap& total_ordering_map);

            ~SegmentExpansionFactorNode();

            //------------------------------------------------------------------
            // Copy & Move constructors/assignments
            //------------------------------------------------------------------

            SegmentExpansionFactorNode(
                const SegmentExpansionFactorNode& factor_node);

            SegmentExpansionFactorNode&
            operator=(const SegmentExpansionFactorNode& factor_node);

            SegmentExpansionFactorNode(SegmentExpansionFactorNode&&) = default;

            SegmentExpansionFactorNode&
            operator=(SegmentExpansionFactorNode&&) = default;

            //------------------------------------------------------------------
            // Static functions
            //------------------------------------------------------------------

            /**
             * Adds an identifier to the label to indicate this is a
             * segment expansion node.
             *
             * @param original_label: original label
             *
             * @return Stamped label
             */
            static std::string compose_label(const std::string& original_label);

            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            bool
            set_incoming_message_from(const MsgNodePtr& source_node_template,
                                      int source_time_step,
                                      int target_time_step,
                                      const Tensor3& message,
                                      Direction direction) override;

            /**
             * Clears messages and beyond a given time step (not inclusive).
             *
             * @param time_step: time step
             */
            void erase_incoming_messages_beyond(int time_step) override;

            /**
             * Updates the probabilities of combinations of left segments for
             * each state value and dependencies of the segment distribution.
             * The messages treated by this factor do not marginalize these
             * dependencies because they have to be preserved for the next
             * time step as the potential function in this factor needs to
             * adjust computations made previously to account for closing and
             * extension of segments. The duration considered in the previous
             * time step needs to be discounted of the incoming message from
             * a previous segment node. If marginalization is performed, the
             * probabilities will not be correct after this process.
             *
             * @param template_target_node: template instance of the node where
             * the message should go to
             * @param template_time_step: time step of this node where to get
             * the incoming messages from. If the template node belongs to the
             * repeatable structure, this information is needed to know which
             * time step to address to retrieve the incoming messages.
             * @param target_time_step: real time step of the target node
             * @param direction: direction of the message passing
             *
             * @return Message
             */
            Tensor3 get_outward_message_to(
                const std::shared_ptr<MessageNode>& template_target_node,
                int template_time_step,
                int target_time_step,
                Direction direction) const override;

            bool is_segment() const override;

            //------------------------------------------------------------------
            // Getters & Setters
            //------------------------------------------------------------------
            int get_timed_node_cardinality() const;

          private:
            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            /**
             * Copies data members from another factor node.
             *
             * @param node: other factor node
             */
            void copy_node(const SegmentExpansionFactorNode& node);

            /**
             * Get message passed from one segment node to another
             *
             * @param template_target_node: template instance of the node where
             * the message should go to
             * @param template_time_step: time step of this node where to get
             * the incoming messages from. If the template node belongs to the
             * repeatable structure, this information is needed to know which
             * time step to address to retrieve the incoming messages.
             * @param direction: direction of the message passing
             * @return
             */
            Tensor3 get_message_between_segments(
                const std::shared_ptr<MessageNode>& template_target_node,
                int template_time_step,
                Direction direction) const;

            /**
             * Expand segment probabilities in one more time step.
             *
             * @param template_time_step: time step of the template node
             *
             * @return Expanded segment probabilities
             */
            Tensor3 expand_segment(int template_time_step) const;

            /**
             * Get message passed from the segment to one of the duration
             * distribution dependencies
             *
             * @param template_target_node: template instance of the node where
             * the message should go to
             * @param template_time_step: time step of this node where to get
             * the incoming messages from. If the template node belongs to the
             * repeatable structure, this information is needed to know which
             * time step to address to retrieve the incoming messages.
             * @param direction: direction of the message passing
             * @return
             */
            Tensor3 get_message_out_of_segments(
                const std::shared_ptr<MessageNode>& template_target_node,
                int template_time_step,
                Direction direction) const;

            /**
             * Gets a tensor with indexing vectors for a segment duration.
             *
             * @param target_node_label: label of the node to which messages
             * are being computed
             * @param template_time_step: time step of this factor node
             * template
             *
             * @return Indexing tensor
             */
            Tensor3 get_indexing_tensor(const std::string& target_node_label,
                                        int template_time_step) const;

            /**
             * Compute and store discounting factors for a segment at a given
             * time step.
             *
             * @param time_step: time step of the timed node of a segment
             */
            void update_discount_factors(int time_step) const;

            //------------------------------------------------------------------
            // Data members
            //------------------------------------------------------------------

            // Order of the duration and transition dependencies. Segment
            // messages will contain a row for each combination of duration
            // and transition dependencies' values.
            CPD::TableOrderingMap total_ordering_map;

            std::unordered_map<int, Tensor3>
                incoming_last_segment_messages_per_time_slice;

            std::unordered_map<int, Tensor3>
                incoming_next_segment_messages_per_time_slice;

            // These discount factors will be updated at every time step.
            // They will be used to compute the probabilities of several
            // combinations of past segment configurations without having to
            // loop over previous time steps.
            mutable Eigen::MatrixXd closing_segment_discount;

            mutable Eigen::MatrixXd extended_segment_discount;

            int timed_node_cardinality = 1;

            int transition_total_cardinality = 1;
        };

    } // namespace model
} // namespace tomcat
