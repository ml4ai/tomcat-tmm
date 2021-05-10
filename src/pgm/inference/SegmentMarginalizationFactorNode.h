#pragma once

#include <utility>

#include "pgm/inference/FactorNode.h"

namespace tomcat {
    namespace model {

        /**
         * Represents a factor node that incorporate state messages to
         * segments in backward passing and marginalizes segments out in
         * forward passing.
         */
        class SegmentMarginalizationFactorNode : public FactorNode {
          public:
            //------------------------------------------------------------------
            // Constructors & Destructor
            //------------------------------------------------------------------

            /**
             * Creates an instance of the segment marginalization factor node.
             *
             * @param label: label of the node that gives raise to segments
             * @param time_step: time step of this template node
             * @param num_segment_message_rows: number of rows in the
             * messages that flow through the expansion and transition segment
             * factor nodes related to the node for which the marginalization
             * is performed
             * @param timed_node_label: label of the node that gives raise to
             * segments
             */
            SegmentMarginalizationFactorNode(
                const std::string& label,
                int time_step,
                int num_segment_message_rows,
                const std::string& timed_node_label);

            ~SegmentMarginalizationFactorNode();

            //------------------------------------------------------------------
            // Copy & Move constructors/assignments
            //------------------------------------------------------------------

            SegmentMarginalizationFactorNode(
                const SegmentMarginalizationFactorNode& factor_node);

            SegmentMarginalizationFactorNode&
            operator=(const SegmentMarginalizationFactorNode& factor_node);

            SegmentMarginalizationFactorNode(
                SegmentMarginalizationFactorNode&&) = default;

            SegmentMarginalizationFactorNode&
            operator=(SegmentMarginalizationFactorNode&&) = default;

            //------------------------------------------------------------------
            // Static functions
            //------------------------------------------------------------------

            /**
             * Adds an identifier to the label to indicate this is a
             * segment marginalization node.
             *
             * @param original_label: original label
             *
             * @return Stamped label
             */
            static std::string compose_label(const std::string& original_label);

            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            /**
             * Marginalizes out segments if the message is passed to a
             * non-segment node, otherwise, replicates the incoming message
             * to all of the segments existent in the factor node'' time step.
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

          private:
            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            /**
             * Copies data members from another factor node.
             *
             * @param node: other factor node
             */
            void copy_node(const SegmentMarginalizationFactorNode& node);

            /**
             * Marginalizes segments out and return messages for a timer
             * controlled node.
             *
             * @param template_time_step: time step of this node where to get
             * the incoming messages from. If the template node belongs to the
             * repeatable structure, this information is needed to know which
             * time step to address to retrieve the incoming messages.
             *
             * @return Message
             */
            Tensor3 marginalize_segments_out(int template_time_step) const;

            /**
             * Expands incoming messages from a timer controlled node to a
             * segment node at a given time step.
             *
             * @param template_time_step: time step of this node where to get
             * the incoming messages from. If the template node belongs to the
             * repeatable structure, this information is needed to know which
             * time step to address to retrieve the incoming messages.
             *
             * @return Message
             */
            Tensor3 expand_incoming_message(int template_time_step) const;

            //------------------------------------------------------------------
            // Data members
            //------------------------------------------------------------------

            int num_segment_message_rows = 1;

            std::string timed_node_label;
        };

    } // namespace model
} // namespace tomcat
