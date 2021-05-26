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
            const CPD::TableOrderingMap& total_ordering_map)
            : FactorNode(compose_label(label),
                         time_step,
                         probability_table,
                         transition_ordering_map,
                         ""),
              total_ordering_map(total_ordering_map) {

            for (const auto& [node_label, indexing_scheme] :
                 transition_ordering_map) {
                if (indexing_scheme.order == 0) {
                    this->timed_node_cardinality = indexing_scheme.cardinality;
                    this->timed_node_label = node_label;
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
        // Static functions
        //----------------------------------------------------------------------
        std::string SegmentTransitionFactorNode::compose_label(
            const std::string& original_label) {
            return "segT:" + original_label;
        }

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        void SegmentTransitionFactorNode::copy_node(
            const SegmentTransitionFactorNode& node) {
            FactorNode::copy_node(node);
            this->total_ordering_map = node.total_ordering_map;
            this->timed_node_cardinality = node.timed_node_cardinality;
        }

        bool SegmentTransitionFactorNode::set_incoming_message_from(
            const MsgNodePtr& source_node_template,
            int source_time_step,
            int target_time_step,
            const Tensor3& message,
            Direction direction) {

            bool changed = false;
            if (!message.is_empty()) {
                this->max_time_step_stored =
                    max(this->max_time_step_stored, target_time_step);

                if (source_node_template->is_segment() &&
                    direction == Direction::forward) {

                    if (EXISTS(
                            target_time_step,
                            this->incoming_last_segment_messages_per_time_slice)) {
                        const Tensor3& curr_msg =
                            this->incoming_last_segment_messages_per_time_slice
                                .at(target_time_step);
                        changed = !curr_msg.equals(message);
                    }
                    else {
                        changed = true;
                    }

                    this->incoming_last_segment_messages_per_time_slice
                        [target_time_step] = message;
                }
                else if (source_node_template->is_segment() &&
                         direction == Direction::backwards) {

                    if (EXISTS(
                            target_time_step,
                            this->incoming_next_segment_messages_per_time_slice)) {
                        const Tensor3& curr_msg =
                            this->incoming_next_segment_messages_per_time_slice
                                .at(target_time_step);
                        changed = !curr_msg.equals(message);
                    }
                    else {
                        changed = true;
                    }

                    this->incoming_next_segment_messages_per_time_slice
                        [target_time_step] = message;
                }
                else {
                    // It does not happen. This node is never connected to a
                    // non-segment node.
                }
            }

            return changed;
        }

        void SegmentTransitionFactorNode::erase_incoming_messages_beyond(
            int time_step) {
            for (int t = time_step + 1; t <= this->max_time_step_stored; t++) {
                this->incoming_messages_per_time_slice.erase(t);
                this->incoming_next_segment_messages_per_time_slice.erase(t);
                this->incoming_last_segment_messages_per_time_slice.erase(t);
            }
            this->max_time_step_stored =
                min(this->max_time_step_stored, time_step);
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
                // Does not happen. This factor node is created automatically
                // by a factor graph when dealing with EDHMM models. There
                // won't be any non-segment node linked to it.
            }

            return output_message;
        }

        Tensor3 SegmentTransitionFactorNode::get_message_between_segments(
            const std::shared_ptr<MessageNode>& template_target_node,
            int template_time_step,
            Direction direction) const {

            Tensor3 output_message;

            if (direction == Direction::forward) {
                Tensor3 indexing_tensor =
                    this->get_indexing_tensor(template_time_step, false);

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
                Tensor3 indexing_tensor =
                    this->get_indexing_tensor(template_time_step, true);

                const auto& table =
                    this->original_potential.node_label_to_rotated_potential
                        .at(this->timed_node_label)
                        .probability_table;

                output_message = Tensor3::dot(indexing_tensor, table);

                Tensor3 open_segment_message =
                    this->extract_open_segment_message(
                        this->incoming_next_segment_messages_per_time_slice.at(
                            template_time_step));

                output_message.hstack(open_segment_message);
            }

            return output_message;
        }

        Tensor3 SegmentTransitionFactorNode::get_indexing_tensor(
            int template_time_step, bool to_expansion_factor) const {
            // Put the incoming messages in order according to the indexing
            // scheme of the potential function.

            int num_dependencies =
                this->original_potential.potential.ordering_map.size();
            vector<Tensor3> indexes(num_dependencies);

            int num_data_points =
                this->incoming_last_segment_messages_per_time_slice
                    .at(template_time_step)
                    .get_shape()[0];
            int rows = this->incoming_last_segment_messages_per_time_slice
                           .at(template_time_step)
                           .get_shape()[1];

            for (int row = 0; row < rows; row++) {
                for (const auto& [node_label, indexing_scheme] :
                     this->total_ordering_map) {
                    if (node_label == this->timed_node_label) {
                        if (to_expansion_factor) {
                            indexes[indexing_scheme.order] =
                                this->extract_closed_segment_message(
                                    this->incoming_next_segment_messages_per_time_slice
                                        .at(template_time_step));
                        }
                        else {
                            indexes[indexing_scheme.order] =
                                this->extract_closed_segment_message(
                                    this->incoming_last_segment_messages_per_time_slice
                                        .at(template_time_step));
                        }
                    }
                    else {
                        if (indexes[indexing_scheme.order].is_empty()) {
                            indexes[indexing_scheme.order] =
                                Tensor3::zeros(num_data_points,
                                               rows,
                                               indexing_scheme.cardinality);
                        }

                        int value =
                            (row /
                             indexing_scheme.right_cumulative_cardinality) %
                            indexing_scheme.cardinality;

                        Eigen::VectorXd binary_value =
                            Eigen::VectorXd::Zero(indexing_scheme.cardinality);
                        binary_value(value) = 1;

                        for (int d = 0; d < num_data_points; d++) {
                            indexes[indexing_scheme.order].row(d, row) =
                                binary_value;
                        }
                    }
                }
            }

            return this->get_cartesian_tensor(indexes);
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
