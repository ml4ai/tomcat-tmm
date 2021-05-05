#include "SegmentTransitionFactorNode.h"

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        SegmentTransitionFactorNode::SegmentTransitionFactorNode(
            const string& label,
            int time_step,
            const Eigen::MatrixXd& probability_table,
            const CPD::TableOrderingMap& transition_ordering_map,
            const CPD::TableOrderingMap& duration_ordering_map)
            : FactorNode(label,
                         time_step,
                         probability_table,
                         transition_ordering_map,
                         ""),
              duration_ordering_map(duration_ordering_map) {

            for (const auto& [label, indexing_scheme] :
                 transition_ordering_map) {
                if (indexing_scheme.order == 0) {
                    this->timed_node_cardinality = indexing_scheme.cardinality;
                    break;
                }
            }
        }

        SegmentTransitionFactorNode::~SegmentTransitionFactorNode() {}

        //----------------------------------------------------------------------
        // Copy & Move constructors/assignments
        //----------------------------------------------------------------------
        SegmentTransitionFactorNode::SegmentTransitionFactorNode(
            const SegmentTransitionFactorNode& factor_node) {
            this->copy_node(factor_node);
        }

        SegmentTransitionFactorNode& SegmentTransitionFactorNode::operator=(
            const SegmentTransitionFactorNode& factor_node) {
            this->copy_node(factor_node);
            return *this;
        }

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------

        void SegmentTransitionFactorNode::copy_node(
            const SegmentTransitionFactorNode& node) {
            FactorNode::copy_node(node);
            this->duration_ordering_map = node.duration_ordering_map;
            this->timed_node_cardinality = node.timed_node_cardinality;
        }

        bool SegmentTransitionFactorNode::is_segment() const { return true; }

        Tensor3 SegmentTransitionFactorNode::get_outward_message_to(
            const shared_ptr<MessageNode>& template_target_node,
            int template_time_step,
            int target_time_step,
            Direction direction) const {

            Tensor3 output_message;

            if (template_target_node->is_segment()) {
                output_message = this->get_message_between_segments(
                    template_target_node, template_time_step, direction);
            }
            else {
                output_message = this->get_message_out_of_segments(
                    template_target_node, template_time_step, direction);
            }

            return output_message;
        }

        Tensor3 SegmentTransitionFactorNode::get_message_between_segments(
            const std::shared_ptr<MessageNode>& template_target_node,
            int template_time_step,
            Direction direction) const {

            Tensor3 output_message;

            if (direction == Direction::forward) {
                Tensor3 indexing_tensor = this->get_indexing_tensor(
                    template_target_node->get_label(), template_time_step);

                output_message = Tensor3::dot(
                    indexing_tensor,
                    this->original_potential.potential.probability_table);

                Tensor3 open_segment_message =
                    this->extract_open_segment_message(
                        this->incoming_last_segment_messages_per_time_slice.at(
                            template_time_step));

                output_message.hstack(open_segment_message);
            }
            else {
            }

            return output_message;
        }

        Tensor3 SegmentTransitionFactorNode::get_message_out_of_segments(
            const std::shared_ptr<MessageNode>& template_target_node,
            int template_time_step,
            Direction direction) const {

            Tensor3 output_message;

            if (EXISTS(template_target_node->get_label(),
                       this->duration_ordering_map)) {
                // If the node is also a parent of the segment duration
                // distribution, its marginal will be computed by the
                // expansion factor node.
                int num_data_points =
                    this->incoming_last_segment_messages_per_time_slice
                        .at(template_time_step)
                        .get_shape()[0];
                int cardinality =
                    this->original_potential.potential.ordering_map
                        .at(template_target_node->get_label())
                        .cardinality;
                output_message = Tensor3::ones(1, num_data_points, cardinality);
            }
            else {
                Tensor3 indexing_tensor = this->get_indexing_tensor(
                    template_target_node->get_label(), template_time_step);

                const auto& table =
                    this->original_potential.node_label_to_rotated_potential
                        .at(template_target_node->get_label())
                        .probability_table;

                Tensor3 segment_transition_msg =
                    Tensor3::dot(indexing_tensor, table)
                        .sum_rows()
                        .reshape(
                            1, indexing_tensor.get_shape()[0], table.cols());

                // Even though the node to which the message is being
                // computed for is only a parent of the transition
                // distribution, we must account with a uniform term for the
                // fact that no transitions occur.
                Tensor3 segment_extension_msg =
                    (this->extract_open_segment_message(
                         this->incoming_next_segment_messages_per_time_slice.at(
                             template_time_step)) *
                     this->extract_open_segment_message(
                         this->incoming_last_segment_messages_per_time_slice.at(
                             template_time_step)))
                        .sum_rows()
                        .sum_cols()
                        .repeat(table.cols(), 2);

                output_message = segment_transition_msg + segment_extension_msg;
            }

            return output_message;
        }

        Tensor3 SegmentTransitionFactorNode::get_indexing_tensor(
            const string& target_node_label, int template_time_step) const {
            // Put the incoming messages in order according to the indexing
            // scheme of the potential function.

            if (!EXISTS(template_time_step,
                        this->incoming_messages_per_time_slice)) {
                // The timed controlled node is the only dependent of the
                // transition distribution. Get it from the segment message.
                return this->extract_closed_segment_message(
                    this->incoming_last_segment_messages_per_time_slice.at(
                        template_time_step));
            }

            const MessageContainer& message_container =
                this->incoming_messages_per_time_slice.at(template_time_step);
            vector<Tensor3> incoming_messages(message_container.size() + 1);
            incoming_messages[0] = this->extract_closed_segment_message(
                this->incoming_last_segment_messages_per_time_slice.at(
                    template_time_step));
            int num_rows = incoming_messages[0].get_shape()[1];

            for (const auto& [incoming_node_name, incoming_message] :
                 message_container.node_name_to_messages) {
                auto [incoming_node_label, incoming_node_time_step] =
                    MessageNode::strip(incoming_node_name);

                const auto& indexing_scheme =
                    this->original_potential.potential.ordering_map.at(
                        incoming_node_label);

                int idx = indexing_scheme.order;

                if (target_node_label == incoming_node_label) {
                    // We will used the rotated potential in this case where
                    // the target node goes to the potential column and is
                    // replaced by the next state, which is the closed
                    // segment portion of the incoming message from the
                    // output segment.
                    incoming_messages[idx] =
                        this->extract_closed_segment_message(
                            this->incoming_next_segment_messages_per_time_slice
                                .at(template_time_step));
                }
                else {
                    int num_data_points = incoming_message.get_shape()[1];
                    int cardinality = incoming_message.get_shape()[2];
                    if (EXISTS(incoming_node_label,
                               this->duration_ordering_map)) {
                        // Fill a tensor with the assignment of the given
                        // node in the respective row in the segment indexing
                        // scheme.
                        Tensor3 identity = Tensor3::zeros(
                            num_data_points, num_rows, cardinality);
                        int rcc =
                            this->duration_ordering_map.at(incoming_node_label)
                                .right_cumulative_cardinality;
                        int col = -1;
                        for (int row = 0; row < num_rows; row++) {
                            if (row % rcc == 0) {
                                col++;
                            }
                            for (int d = 0; d < num_data_points; d++) {
                                identity(d, row, col) = 1;
                            }
                        }

                        incoming_messages[idx] = identity;
                    }
                    else {
                        // Move messages per data point to the depth axis and
                        // repeat the message along the row axis by the
                        // number of combinations the incoming segment
                        // message has.
                        incoming_messages[idx] =
                            incoming_message
                                .reshape(num_data_points, 1, cardinality)
                                .repeat(num_rows, 1);
                    }
                }
            }

            return this->get_cartesian_tensor(incoming_messages);
        }

        Tensor3 SegmentTransitionFactorNode::extract_closed_segment_message(
            const Tensor3& full_segment_msg) const {
            return full_segment_msg.slice(0, this->timed_node_cardinality, 2);
        }

        Tensor3 SegmentTransitionFactorNode::extract_open_segment_message(
            const Tensor3& full_segment_msg) const {
            return full_segment_msg.slice(
                this->timed_node_cardinality, Tensor3::ALL, 2);
        }

        //----------------------------------------------------------------------
        // Getters & Setters
        //----------------------------------------------------------------------

    } // namespace model
} // namespace tomcat
