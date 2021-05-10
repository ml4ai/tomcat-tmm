#include "SegmentMarginalizationFactorNode.h"

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        SegmentMarginalizationFactorNode::SegmentMarginalizationFactorNode(
            const string& label,
            int time_step,
            int num_segment_message_rows,
            const string& timed_node_label)
            : FactorNode(compose_label(label), time_step),
              num_segment_message_rows(num_segment_message_rows),
              timed_node_label(timed_node_label) {}

        SegmentMarginalizationFactorNode::~SegmentMarginalizationFactorNode() {}

        //----------------------------------------------------------------------
        // Copy & Move constructors/assignments
        //----------------------------------------------------------------------
        SegmentMarginalizationFactorNode::SegmentMarginalizationFactorNode(
            const SegmentMarginalizationFactorNode& factor_node) {
            this->copy_node(factor_node);
        }

        SegmentMarginalizationFactorNode&
        SegmentMarginalizationFactorNode::operator=(
            const SegmentMarginalizationFactorNode& factor_node) {
            this->copy_node(factor_node);
            return *this;
        }

        //----------------------------------------------------------------------
        // Static functions
        //----------------------------------------------------------------------
        std::string SegmentMarginalizationFactorNode::compose_label(
            const std::string& original_label) {
            return "segM:" + original_label;
        }

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        void SegmentMarginalizationFactorNode::copy_node(
            const SegmentMarginalizationFactorNode& node) {
            FactorNode::copy_node(node);
        }

        bool SegmentMarginalizationFactorNode::is_segment() const {
            // It has a special potential function but messages from this
            // node are not treated as segments. They will be saved under the
            // regular incoming messages list in variable nodes that link to
            // this factor.
            return false;
        }

        Tensor3 SegmentMarginalizationFactorNode::get_outward_message_to(
            const shared_ptr<MessageNode>& template_target_node,
            int template_time_step,
            int target_time_step,
            Direction direction) const {

            Tensor3 output_message;

            if (template_target_node->get_label() == this->timed_node_label) {
                output_message =
                    this->marginalize_segments_out(template_time_step);
            }
            else {
                output_message =
                    this->expand_incoming_message(template_time_step);
            }

            return output_message;
        }

        Tensor3 SegmentMarginalizationFactorNode::marginalize_segments_out(
            int template_time_step) const {

            Tensor3 message_from_segment;
            for (const auto& [incoming_node_name, incoming_message] :
                 this->incoming_messages_per_time_slice.at(template_time_step)
                     .node_name_to_messages) {

                auto [incoming_node_label, incoming_node_time_step] =
                    MessageNode::strip(incoming_node_name);

                if (incoming_node_label == this->timed_node_label) {
                    continue;
                }

                message_from_segment = incoming_message;
                break;
            }

            int num_segment_combinations = template_time_step + 1;
            int cardinality =
                message_from_segment.get_shape()[2] / num_segment_combinations;

            Tensor3 marginalization_function =
                Tensor3::eye(message_from_segment.get_shape()[0], cardinality)
                    .repeat(num_segment_combinations, 1);

            return Tensor3::dot(message_from_segment, marginalization_function)
                .sum_rows()
                .reshape(1, message_from_segment.get_shape()[0], cardinality);
        }

        Tensor3 SegmentMarginalizationFactorNode::expand_incoming_message(
            int template_time_step) const {

            Tensor3 message_from_timed_node;
            for (const auto& [incoming_node_name, incoming_message] :
                 this->incoming_messages_per_time_slice.at(template_time_step)
                     .node_name_to_messages) {

                auto [incoming_node_label, incoming_node_time_step] =
                    MessageNode::strip(incoming_node_name);

                if (incoming_node_label == this->timed_node_label) {
                    message_from_timed_node = incoming_message;
                    break;
                }
            }

            message_from_timed_node = message_from_timed_node.reshape(
                message_from_timed_node.get_shape()[1],
                1,
                message_from_timed_node.get_shape()[2]);

            int num_segment_combinations = template_time_step + 1;
            return message_from_timed_node.repeat(num_segment_combinations, 2)
                .repeat(this->num_segment_message_rows, 1);
        }

    } // namespace model
} // namespace tomcat
