#include "SegmentExpansionFactorNode.h"

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        SegmentExpansionFactorNode::SegmentExpansionFactorNode(
            const string& label,
            int time_step,
            const DistributionPtrVec& duration_distributions,
            const CPD::TableOrderingMap& duration_ordering_map,
            const CPD::TableOrderingMap& total_ordering_map)
            : FactorNode(compose_label(label),
                         time_step,
                         duration_distributions,
                         duration_ordering_map,
                         "",
                         false),
              total_ordering_map(total_ordering_map) {

            for (const auto& [label, indexing_scheme] : duration_ordering_map) {
                if (indexing_scheme.order == 0) {
                    this->timed_node_cardinality = indexing_scheme.cardinality;
                    break;
                }
            }

            // We store the number of combinations of the dependencies of the
            // transition distribution that are not dependencies of the
            // duration distribution. We need this information to replicate the
            // discount factors to match the number of possible combinations
            // between duration and transition dependencies' values.
            int num_duration_dependencies = duration_ordering_map.size();
            for (const auto& [node_label, indexing_scheme] :
                 total_ordering_map) {
                if (indexing_scheme.order == num_duration_dependencies) {
                    this->transition_total_cardinality =
                        indexing_scheme.cardinality *
                        indexing_scheme.right_cumulative_cardinality;
                    break;
                }
            }
        }

        SegmentExpansionFactorNode::~SegmentExpansionFactorNode() {}

        //----------------------------------------------------------------------
        // Copy & Move constructors/assignments
        //----------------------------------------------------------------------
        SegmentExpansionFactorNode::SegmentExpansionFactorNode(
            const SegmentExpansionFactorNode& factor_node) {
            this->copy_node(factor_node);
        }

        SegmentExpansionFactorNode& SegmentExpansionFactorNode::operator=(
            const SegmentExpansionFactorNode& factor_node) {
            this->copy_node(factor_node);
            return *this;
        }

        //----------------------------------------------------------------------
        // Static functions
        //----------------------------------------------------------------------
        std::string SegmentExpansionFactorNode::compose_label(
            const std::string& original_label) {
            return "segE:" + original_label;
        }

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        void SegmentExpansionFactorNode::copy_node(
            const SegmentExpansionFactorNode& node) {
            FactorNode::copy_node(node);
            this->closing_segment_discount = node.closing_segment_discount;
            this->extended_segment_discount = node.extended_segment_discount;
            this->timed_node_cardinality = node.timed_node_cardinality;
            this->total_ordering_map = node.total_ordering_map;
        }

        bool SegmentExpansionFactorNode::set_incoming_message_from(
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
                    if (EXISTS(target_time_step,
                               this->incoming_messages_per_time_slice)) {
                        const Tensor3& curr_msg =
                            this->incoming_messages_per_time_slice
                                .at(target_time_step)
                                .get_message_for(
                                    source_node_template->get_label(),
                                    source_time_step);
                        changed = !curr_msg.equals(message);
                    }
                    else {
                        changed = true;
                    }

                    this->incoming_messages_per_time_slice[target_time_step]
                        .set_message_for(source_node_template->get_label(),
                                         source_time_step,
                                         message);
                }
            }

            return changed;
        }

        void SegmentExpansionFactorNode::erase_incoming_messages_beyond(
            int time_step) {
            for (int t = time_step + 1; t <= this->max_time_step_stored; t++) {
                this->incoming_messages_per_time_slice.erase(t);
                this->incoming_next_segment_messages_per_time_slice.erase(t);
                this->incoming_last_segment_messages_per_time_slice.erase(t);
            }
            this->max_time_step_stored =
                min(this->max_time_step_stored, time_step);
        }

        bool SegmentExpansionFactorNode::is_segment() const { return true; }

        Tensor3 SegmentExpansionFactorNode::get_outward_message_to(
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

        Tensor3 SegmentExpansionFactorNode::get_message_between_segments(
            const std::shared_ptr<MessageNode>& template_target_node,
            int template_time_step,
            Direction direction) const {

            Tensor3 output_message;

            if (direction == Direction::forward) {
                this->update_discount_factors(template_time_step);

                // Weighing everything with the messages from the
                // dependencies of the duration distribution.
                Tensor3 indexing_tensor = this->get_indexing_tensor(
                    template_target_node->get_label(), template_time_step);
                indexing_tensor.transpose_matrices();

                if (template_time_step == 0 &&
                    !EXISTS(
                        0,
                        this->incoming_last_segment_messages_per_time_slice)) {

                    // Prior
                    output_message =
                        Tensor3::ones(indexing_tensor.get_shape()[0],
                                      indexing_tensor.get_shape()[1],
                                      this->timed_node_cardinality);
                }
                else {
                    output_message = this->expand_segment(template_time_step);
                }

                output_message =
                    output_message.mult_colwise_broadcasting(indexing_tensor);
            }
            else {
                // Messages won't be passed backwards between segments.
            }

            return output_message;
        }

        Tensor3 SegmentExpansionFactorNode::get_indexing_tensor(
            const string& target_node_label, int template_time_step) const {
            // Put the incoming messages in order according to the indexing
            // scheme of the potential function.

            Tensor3 indexing_tensor;

            if (!EXISTS(template_time_step,
                        this->incoming_messages_per_time_slice)) {
                int num_data_points =
                    this->incoming_last_segment_messages_per_time_slice
                        .at(template_time_step)
                        .get_shape()[0];
                indexing_tensor = Tensor3::ones(num_data_points, 1, 1);
            }
            else {
                const MessageContainer& message_container =
                    this->incoming_messages_per_time_slice.at(
                        template_time_step);
                vector<Tensor3> incoming_messages(message_container.size());

                for (const auto& [incoming_node_name, incoming_message] :
                     message_container.node_name_to_messages) {
                    auto [incoming_node_label, incoming_node_time_step] =
                        MessageNode::strip(incoming_node_name);
                    // There's only one non-segment node sending message to this
                    // factor, which is a joint node (duration + transition
                    // dependencies). The message from that node is the indexing
                    // tensor. We just need to reshape it so that different data
                    // points are distributed across the depth of the tensor.

                    // The first index in the distribution belongs to a time
                    // controlled node which is not a neighbor of this factor
                    // node. Therefore, we subtract 1 from the order to find the
                    // right index.
                    indexing_tensor = incoming_message.reshape(
                        incoming_message.get_shape()[1],
                        1,
                        incoming_message.get_shape()[2]);
                    break;
                }
            }

            return indexing_tensor;
        }

        Tensor3 SegmentExpansionFactorNode::expand_segment(
            int template_time_step) const {
            Tensor3 message_from_last_segment =
                this->incoming_last_segment_messages_per_time_slice.at(
                    template_time_step);

            // Computing the probabilities of closing previous
            // segment configurations.
            Tensor3 closed_weighted_segments =
                message_from_last_segment * this->closing_segment_discount;

            // This tensor works as a function to marginalize
            // segments out
            Eigen::MatrixXd segment_marginalizer =
                Eigen::MatrixXd::Identity(this->timed_node_cardinality,
                                          this->timed_node_cardinality)
                    .replicate(template_time_step, 1);

            Tensor3 closing_segment_probs =
                Tensor3::dot(closed_weighted_segments, segment_marginalizer);

            // Computing the probability of extending previous
            // segment configurations.
            Tensor3 extended_weighted_segments =
                message_from_last_segment * this->extended_segment_discount;

            Tensor3 expanded_segment = closing_segment_probs;
            expanded_segment.hstack(extended_weighted_segments);

            return expanded_segment;
        }

        Tensor3 SegmentExpansionFactorNode::get_message_out_of_segments(
            const std::shared_ptr<MessageNode>& template_target_node,
            int template_time_step,
            Direction direction) const {

            // Message from transition factor
            Tensor3 segment_table =
                this->incoming_next_segment_messages_per_time_slice.at(
                    template_time_step);

            if (EXISTS(template_time_step,
                       this->incoming_last_segment_messages_per_time_slice)) {
                segment_table =
                    segment_table * this->expand_segment(template_time_step);
            }

            // Marginalize segments and states out
            Tensor3 output_message = segment_table.sum_cols();
            output_message.transpose_matrices();
            output_message =
                output_message.reshape(1,
                                       output_message.get_shape()[0],
                                       output_message.get_shape()[2]);
            output_message.normalize_rows();

            return output_message;
        }

        void SegmentExpansionFactorNode::update_discount_factors(
            int time_step) const {
            // Update discount factors for closing the previous segments for
            // every possible assignment of the time controlled
            // node.
            //
            // The probability of closing previous segments is the
            // probability of closing any of the past previous
            // configurations, multiplied by the pdf of duration equality
            // (when a segment is closed, the duration is equal to a number)
            // and divided by the previous duration cdf (when a segment is
            // open, we assume that the duration is bigger or equals to a
            // number).

            if (this->closing_segment_discount.cols() <
                time_step * this->timed_node_cardinality) {
                // Discounts have not been initialized yet. In case this
                // factor node is part of the repeatable structure of the
                // factor graph, we need to initialize until the first
                // time step. We could get the computation already
                // performed for the previous time steps but since the
                // repeatable structure is at the time step 2,
                // recomputing the discounts for time step 0 and 1 here
                // won't cost much.
                int num_distributions =
                    this->original_potential.potential.distributions.size();

                // The number of distributions is given by the cartesian
                // product of the cardinality of the timed node and the
                // other dependencies of the duration distribution. We
                // will store the parent nodes in the rows of the
                // discount table and the joint (timed nodes, duration)
                // in the columns of the matrix.
                int duration_rows =
                    num_distributions / this->timed_node_cardinality;
                int total_rows =
                    duration_rows * this->transition_total_cardinality;
                int cols = timed_node_cardinality * time_step;
                int previous_cols = this->closing_segment_discount.cols();

                this->closing_segment_discount.conservativeResize(total_rows,
                                                                  cols);
                this->extended_segment_discount.conservativeResize(total_rows,
                                                                   cols);

                for (int t = previous_cols / this->timed_node_cardinality;
                     t < time_step;
                     t++) {
                    // After initialization of the discount matrix. This
                    // will comprehend just one time step.
                    int i = 0;
                    for (const auto& distribution :
                         this->original_potential.potential.distributions) {

                        int row = i % duration_rows;
                        int col = (t * this->timed_node_cardinality) +
                                  i / duration_rows;
                        double cdf = distribution->get_cdf(t - 1, true);
                        double value_closing_segment = 0;
                        double value_opening_segment = 0;

                        if (cdf != 0) {
                            value_closing_segment =
                                distribution->get_pdf(t) / cdf;
                            value_opening_segment =
                                distribution->get_cdf(t, true) / cdf;
                        }

                        // Values replicated for rows that represent
                        // combinations of values from dependencies of the
                        // segment transition only.
                        for (int j = 0; j < this->transition_total_cardinality;
                             j++) {
                            this->closing_segment_discount(
                                j + row * this->transition_total_cardinality,
                                col) = value_closing_segment;
                            this->extended_segment_discount(
                                j + row * this->transition_total_cardinality,
                                col) = value_opening_segment;
                        }

                        i += 1;
                    }
                }
            }
        }

        //----------------------------------------------------------------------
        // Getters & Setters
        //----------------------------------------------------------------------
        int SegmentExpansionFactorNode::get_timed_node_cardinality() const {
            return timed_node_cardinality;
        }

    } // namespace model
} // namespace tomcat
