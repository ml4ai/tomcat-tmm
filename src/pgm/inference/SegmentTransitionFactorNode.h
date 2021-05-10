#pragma once

#include <utility>

#include "pgm/inference/FactorNode.h"

namespace tomcat {
    namespace model {

        /**
         * Represents a factor node that incorporate transition probabilities
         * in closed segments because a closed segment happens whenever
         * there's a transition to a different state.
         */
        class SegmentTransitionFactorNode : public FactorNode {
          public:
            //------------------------------------------------------------------
            // Constructors & Destructor
            //------------------------------------------------------------------

            /**
             * Creates factor node to update the probabilities of a closed
             * segment.
             */
            SegmentTransitionFactorNode(
                const std::string& label,
                int time_step,
                const Eigen::MatrixXd& probability_table,
                const CPD::TableOrderingMap& transition_ordering_map,
                const CPD::TableOrderingMap& duration_ordering_map);

            ~SegmentTransitionFactorNode();

            //------------------------------------------------------------------
            // Copy & Move constructors/assignments
            //------------------------------------------------------------------

            SegmentTransitionFactorNode(
                const SegmentTransitionFactorNode& factor_node);

            SegmentTransitionFactorNode&
            operator=(const SegmentTransitionFactorNode& factor_node);

            SegmentTransitionFactorNode(SegmentTransitionFactorNode&&) =
                default;

            SegmentTransitionFactorNode&
            operator=(SegmentTransitionFactorNode&&) = default;

            //------------------------------------------------------------------
            // Static functions
            //------------------------------------------------------------------

            /**
             * Adds an identifier to the label to indicate this is a
             * segment transition node.
             *
             * @param original_label: original label
             *
             * @return Stamped label
             */
            static std::string compose_label(const std::string& original_label);

            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            void
            set_incoming_message_from(const MsgNodePtr& source_node_template,
                                      int source_time_step,
                                      int target_time_step,
                                      const Tensor3& message,
                                      Direction direction) override;

            void erase_incoming_messages_beyond(int time_step) override;

            /**
             * Accounts for the transition probabilities in the closed
             * segment portion of the incoming message from another segment
             * node.
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

          private:
            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            /**
             * Copies data members from another factor node.
             *
             * @param node: other factor node
             */
            void copy_node(const SegmentTransitionFactorNode& node);

            Tensor3 get_message_between_segments(
                const std::shared_ptr<MessageNode>& template_target_node,
                int template_time_step,
                Direction direction) const;

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
             * @param to_expansion_factor: whether the message is being
             * passed backwards to an expansion factor
             *
             * @return Indexing tensor
             */
            Tensor3 get_indexing_tensor(const std::string& target_node_label,
                                        int template_time_step,
                                        bool to_expansion_factor) const;

            /**
             * Get the first portion of the segment message related to the
             * probabilities of closing a previous segment.
             *
             * @param full_segment_msg: segment message
             *
             * @return Closed segment message
             */
            Tensor3 extract_closed_segment_message(
                const Tensor3& full_segment_msg) const;

            /**
             * Get the last portion of the segment message related to the
             * probabilities of extending a previous segment.
             *
             * @param full_segment_msg: segment message
             *
             * @return Open segment message
             */
            Tensor3
            extract_open_segment_message(const Tensor3& full_segment_msg) const;

            //------------------------------------------------------------------
            // Data members
            //------------------------------------------------------------------

            std::unordered_map<int, Tensor3>
                incoming_last_segment_messages_per_time_slice;

            std::unordered_map<int, Tensor3>
                incoming_next_segment_messages_per_time_slice;

            CPD::TableOrderingMap duration_ordering_map;

            int timed_node_cardinality;

            std::string timed_node_label;
        };

    } // namespace model
} // namespace tomcat
