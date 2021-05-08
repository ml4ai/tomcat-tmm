#include "SegmentMarginalizationFactorNode.h"

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        SegmentMarginalizationFactorNode::SegmentMarginalizationFactorNode(
            const string& label, int time_step)
            : FactorNode(compose_label(label), time_step) {}

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
            return true;
        }

        Tensor3 SegmentMarginalizationFactorNode::get_outward_message_to(
            const shared_ptr<MessageNode>& template_target_node,
            int template_time_step,
            int target_time_step,
            Direction direction) const {

            Tensor3 output_message;

            if (template_target_node->is_segment()) {
                output_message =
                    this->expand_incoming_message(template_time_step);
            }
            else {
                output_message =
                    this->marginalize_segments_out(template_time_step);
            }

            return output_message;
        }

        Tensor3 SegmentMarginalizationFactorNode::marginalize_segments_out(
            int template_time_step) const {

            const Tensor3& incoming_message =
                this->incoming_last_segment_messages_per_time_slice.at(
                    template_time_step);

            int num_segment_combinations = template_time_step + 1;
            int cardinality =
                incoming_message.get_shape()[2] / num_segment_combinations;

            Tensor3 marginalization_function =
                Tensor3::eye(incoming_message.get_shape()[0], cardinality)
                    .repeat(num_segment_combinations, 1);

            return Tensor3::dot(incoming_message, marginalization_function)
                .sum_rows()
                .reshape(1, incoming_message.get_shape()[0], cardinality);
        }

        Tensor3 SegmentMarginalizationFactorNode::expand_incoming_message(
            int template_time_step) const {

            Tensor3 incoming_message;
            for (const auto& [node_label, message] :
                 this->incoming_messages_per_time_slice.at(template_time_step)
                     .node_name_to_messages) {
                // This factor node is linked to a single non-segment node only.
                incoming_message = message;
                break;
            }

            int num_segment_combinations = template_time_step + 1;
            return incoming_message.repeat(num_segment_combinations, 2);
        }

    } // namespace model
} // namespace tomcat
