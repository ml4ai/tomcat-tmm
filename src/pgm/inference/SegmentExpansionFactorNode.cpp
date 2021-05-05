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
            const CPD::TableOrderingMap& duration_ordering_map)
            : FactorNode(label,
                         time_step,
                         duration_distributions,
                         duration_ordering_map,
                         "") {

            for (const auto& [label, indexing_scheme] : duration_ordering_map) {
                if (indexing_scheme.order == 0) {
                    this->timed_node_cardinality = indexing_scheme.cardinality;
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
        // Member functions
        //----------------------------------------------------------------------

        void SegmentExpansionFactorNode::copy_node(
            const SegmentExpansionFactorNode& node) {
            FactorNode::copy_node(node);
            this->closing_segment_discount = node.closing_segment_discount;
            this->extended_segment_discount = node.extended_segment_discount;
            this->timed_node_cardinality = node.timed_node_cardinality;
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

                Tensor3 closing_segment_probs = Tensor3::dot(
                    closed_weighted_segments, segment_marginalizer);

                // Computing the probability of extending previous
                // segment configurations.
                Tensor3 extended_weighted_segments =
                    message_from_last_segment * this->extended_segment_discount;

                output_message = closing_segment_probs;
                output_message.hstack(extended_weighted_segments);

                // Weighing everything with the messages from the
                // dependencies of the duration distribution.
                Tensor3 indexing_tensor = this->get_indexing_tensor(
                    template_target_node->get_label(), template_time_step);

                indexing_tensor =
                    indexing_tensor.reshape(indexing_tensor.get_shape()[1],
                                            1,
                                            indexing_tensor.get_shape()[2]);
                indexing_tensor.transpose_matrices();

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

            if (!EXISTS(template_time_step,
                        this->incoming_messages_per_time_slice)) {
                int num_data_points =
                    this->incoming_last_segment_messages_per_time_slice
                        .at(template_time_step)
                        .get_shape()[0];
                return Tensor3::ones(num_data_points, 1, 1);
            }

            const MessageContainer& message_container =
                this->incoming_messages_per_time_slice.at(template_time_step);
            vector<Tensor3> incoming_messages(message_container.size());
            for (const auto& [incoming_node_name, incoming_message] :
                 message_container.node_name_to_messages) {
                auto [incoming_node_label, incoming_node_time_step] =
                    MessageNode::strip(incoming_node_name);

                const auto& indexing_scheme =
                    this->original_potential.potential.ordering_map.at(
                        incoming_node_label);

                // The first index in the distribution belongs to a time
                // controlled node which is not a neighbor of this factor
                // node. Therefore, we subtract 1 from the order to find the
                // right index.
                int idx = indexing_scheme.order - 1;

                if (target_node_label == incoming_node_label) {
                    // If the node we are computing messages to is one of
                    // the indexing nodes of the duration distribution, we
                    // ignore messages from it when computing the message
                    // that goes to it. We can do so by replacing its
                    // outcome message by a uniform one.
                    const auto& shape = incoming_message.get_shape();
                    incoming_messages[idx] =
                        Tensor3::ones(shape.at(0), shape.at(1), shape.at(2));
                }
                else {
                    incoming_messages[idx] = incoming_message;
                }
            }

            return this->get_cartesian_tensor(incoming_messages);
        }

        Tensor3 SegmentExpansionFactorNode::get_message_out_of_segments(
            const std::shared_ptr<MessageNode>& template_target_node,
            int template_time_step,
            Direction direction) const {

            Tensor3 indexing_tensor = this->get_indexing_tensor(
                template_target_node->get_label(), template_time_step);

            Tensor3 message_from_isegment =
                this->incoming_next_segment_messages_per_time_slice.at(
                    template_time_step);

            // Marginalize segments and states out
            message_from_isegment = message_from_isegment.sum_cols();
            const auto& indexing_scheme =
                this->original_potential.potential.ordering_map.at(
                    template_target_node->get_label());

            Eigen::MatrixXd message_matrix =
                Eigen::MatrixXd::Zero(message_from_isegment.get_shape().at(0),
                                      indexing_scheme.cardinality);
            for (int i = 0; i < message_from_isegment.get_shape().at(0); i++) {
                // Indexing tensor contains only  Each row of this
                // matrix
                // contains an indexing vector
                Eigen::MatrixXd rotated_duration_indexing = this->rotate_table(
                    indexing_tensor(0, 0).row(i).transpose(),
                    indexing_scheme.cardinality,
                    indexing_scheme.right_cumulative_cardinality);

                Eigen::MatrixXd rotated_segment_duration = this->rotate_table(
                    message_from_isegment(i, 0),
                    indexing_scheme.cardinality,
                    indexing_scheme.right_cumulative_cardinality);

                message_matrix.row(i) = (rotated_duration_indexing.array() *
                                         rotated_segment_duration.array())
                                            .colwise()
                                            .sum();
            }

            return Tensor3(message_matrix);
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

            if (this->closing_segment_discount.cols() < time_step) {
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
                int rows = num_distributions / this->timed_node_cardinality;
                int cols = timed_node_cardinality * time_step;
                int previous_cols = this->closing_segment_discount.cols();

                this->closing_segment_discount.conservativeResize(rows, cols);
                this->extended_segment_discount.conservativeResize(rows, cols);

                for (int t = previous_cols / this->timed_node_cardinality;
                     t < time_step;
                     t++) {
                    // After initialization of the discount matrix. This
                    // will comprehend just one time step.
                    int i = 0;
                    for (const auto& distribution :
                         this->original_potential.potential.distributions) {

                        int row = i % rows;
                        int col = (t * this->timed_node_cardinality) + i / rows;
                        this->closing_segment_discount(row, col) =
                            distribution->get_pdf(t) /
                            distribution->get_cdf(t - 1, true);

                        this->extended_segment_discount(row, col) =
                            distribution->get_cdf(t, true) /
                            distribution->get_cdf(t - 1, true);

                        i += 1;
                    }
                }
            }
        }

        //----------------------------------------------------------------------
        // Getters & Setters
        //----------------------------------------------------------------------

    } // namespace model
} // namespace tomcat
