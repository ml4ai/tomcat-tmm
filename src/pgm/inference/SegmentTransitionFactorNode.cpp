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
                    this->incoming_last_segment_messages_per_time_slice
                        .at(template_time_step)
                        .slice(this->timed_node_cardinality, Tensor3::ALL, 2);

                output_message.hstack(open_segment_message);
            } else {

            }

            return output_message;
        }

        Tensor3 SegmentTransitionFactorNode::get_message_out_of_segments(
            const std::shared_ptr<MessageNode>& template_target_node,
            int template_time_step,
            Direction direction) const {

            //            Tensor3 indexing_tensor = this->get_indexing_tensor(
            //                template_target_node->get_label(),
            //                template_time_step);
            //
            //            Tensor3 message_from_isegment =
            //                this->incoming_next_segment_messages_per_time_slice.at(
            //                    template_time_step);
            //
            //            // Marginalize segments and states out
            //            message_from_isegment =
            //            message_from_isegment.sum_cols(); const auto&
            //            indexing_scheme =
            //                this->duration_potential.ordering_map.at(
            //                    template_target_node->get_label());
            //
            //            Eigen::MatrixXd message_matrix =
            //                Eigen::MatrixXd::Zero(message_from_isegment.get_shape().at(0),
            //                                      indexing_scheme.cardinality);
            //            for (int i = 0; i <
            //            message_from_isegment.get_shape().at(0); i++) {
            //                // Indexing tensor contains only  Each row of this
            //                // matrix
            //                // contains an indexing vector
            //                Eigen::MatrixXd rotated_duration_indexing =
            //                this->rotate_table(
            //                    indexing_tensor(0, 0).row(i).transpose(),
            //                    indexing_scheme.cardinality,
            //                    indexing_scheme.right_cumulative_cardinality);
            //
            //                Eigen::MatrixXd rotated_segment_duration =
            //                this->rotate_table(
            //                    message_from_isegment(i, 0),
            //                    indexing_scheme.cardinality,
            //                    indexing_scheme.right_cumulative_cardinality);
            //
            //                message_matrix.row(i) =
            //                (rotated_duration_indexing.array() *
            //                                         rotated_segment_duration.array())
            //                                            .colwise()
            //                                            .sum();
            //            }
            //
            //            return Tensor3(message_matrix);
            return Tensor3();
        }

        Tensor3 SegmentTransitionFactorNode::get_indexing_tensor(
            const string& target_node_label, int template_time_step) const {
            // Put the incoming messages in order according to the indexing
            // scheme of the potential function.

            if (!EXISTS(template_time_step,
                this->incoming_messages_per_time_slice)) {
                // The timed controlled node is the only dependent of the
                // transition distribution. Get it from the segment message.
                return this->incoming_last_segment_messages_per_time_slice
                    .at(template_time_step)
                    .slice(0, this->timed_node_cardinality, 2);;
            }

            const MessageContainer& message_container =
                this->incoming_messages_per_time_slice.at(template_time_step);
            vector<Tensor3> incoming_messages(message_container.size());
            incoming_messages[0] =
                this->incoming_last_segment_messages_per_time_slice
                    .at(template_time_step)
                    .slice(0, this->timed_node_cardinality, 2);

            for (const auto& [incoming_node_name, incoming_message] :
                 message_container.node_name_to_messages) {
                auto [incoming_node_label, incoming_node_time_step] =
                    MessageNode::strip(incoming_node_name);

                const auto& indexing_scheme =
                    this->original_potential.potential.ordering_map.at(
                        incoming_node_label);

                int idx = indexing_scheme.order;

                if (target_node_label == incoming_node_label) {
                    // If the node we are computing messages to is one of the
                    // indexing nodes of the duration distribution, we ignore
                    // messages from it when computing the message that goes
                    // to it. We can do so by replacing its outcome message
                    // by a uniform one.
                    //                    const auto& shape =
                    //                    incoming_message.get_shape();
                    //                    incoming_messages[idx] =
                    //                    Tensor3::constant(
                    //                        shape.at(0), shape.at(1),
                    //                        shape.at(2), 1);
                }
                else {
                    int num_data_points = incoming_message.get_shape()[1];
                    int cardinality = incoming_message.get_shape()[2];
                    if (EXISTS(incoming_node_label,
                               this->duration_ordering_map)) {
                        incoming_messages[idx] =
                            Tensor3::eye(num_data_points, cardinality);
                    }
                    else {
                        // Move messages per data point to the depth axis and
                        // repeat the message along the row axis by the
                        // number of combinations the incoming segment
                        // message has.
                        incoming_messages[idx] =
                            incoming_message
                                .reshape(num_data_points, 1, cardinality)
                                .repeat(incoming_messages[0].get_shape()[1], 1);
                    }
                }
            }

            return this->get_cartesian_tensor(incoming_messages);
        }

        //----------------------------------------------------------------------
        // Getters & Setters
        //----------------------------------------------------------------------

    } // namespace model
} // namespace tomcat
