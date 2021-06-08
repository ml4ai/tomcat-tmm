#include "VariableNode.h"

using namespace std;

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        VariableNode::VariableNode(const string& label,
                                   int time_step,
                                   int cardinality)
            : MessageNode(label, time_step), cardinality(cardinality) {}

        VariableNode::VariableNode(const string& label, int time_step)
            : MessageNode(compose_segment_label(label), time_step),
              segment(true) {}

        VariableNode::~VariableNode() {}

        //----------------------------------------------------------------------
        // Copy & Move constructors/assignments
        //----------------------------------------------------------------------
        VariableNode::VariableNode(const VariableNode& node) {
            this->copy_node(node);
        }

        VariableNode& VariableNode::operator=(const VariableNode& node) {
            this->copy_node(node);
            return *this;
        }

        //----------------------------------------------------------------------
        // Static functions
        //----------------------------------------------------------------------
        string
        VariableNode::compose_segment_label(const string& timed_node_label) {
            return "seg:" + timed_node_label;
        }

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        void VariableNode::copy_node(const VariableNode& node) {
            MessageNode::copy_node(node);
            this->cardinality = node.cardinality;
            this->data_per_time_slice = node.data_per_time_slice;
        }

        Tensor3 VariableNode::get_outward_message_to(
            const shared_ptr<MessageNode>& template_target_node,
            int template_time_step,
            int target_time_step,
            Direction direction) const {

            Tensor3 outward_message;
            if (EXISTS(template_time_step, this->data_per_time_slice)) {
                // If there's data for the node, just report the
                // one-hot-encode representation of that data as the
                // message emitted by this node.
                outward_message =
                    Tensor3(this->data_per_time_slice.at(template_time_step));
            }
            else {
                if (EXISTS(template_time_step,
                           this->incoming_messages_per_time_slice)) {
                    MessageContainer message_container =
                        this->incoming_messages_per_time_slice.at(
                            template_time_step);

                    for (const auto& [incoming_node_name, incoming_message] :
                         message_container.node_name_to_messages) {

                        auto [incoming_node_label, incoming_node_time_step] =
                            MessageNode::strip(incoming_node_name);

                        // Only proceeds if the incoming node is not on the list
                        // of nodes to be ignored for the target.
                        if (direction == Direction::backwards &&
                            EXISTS(template_target_node->get_label(),
                                   this->backward_blocking)) {
                            const auto& ignore_set = this->backward_blocking.at(
                                template_target_node->get_label());
                            if (EXISTS(incoming_node_label, ignore_set)) {
                                continue;
                            }
                        }

                        if (incoming_node_name ==
                            MessageNode::get_name(
                                template_target_node->get_label(),
                                target_time_step)) {
                            continue;
                        }

                        if (outward_message.is_empty()) {
                            outward_message = incoming_message;
                        }
                        else {
                            outward_message =
                                outward_message * incoming_message;
                        }
                    }

                    if (!this->segment || template_target_node->is_segment()) {
                        // If the node is a segment node, only normalize if the
                        // message goes to a segment factor. This avoid
                        // normalizing messages to a segment marginalization
                        // factor, which would remove the contribution of the
                        // segment dependencies to the time controlled node
                        // probability.
                        outward_message.normalize_rows();
                    }
                }
            }

            return outward_message;
        }

        bool VariableNode::is_factor() const { return false; }

        bool VariableNode::is_segment() const { return this->segment; }

        Eigen::MatrixXd VariableNode::get_marginal_at(int time_step,
                                                      bool normalized) const {
            Eigen::MatrixXd marginal;

            for (const auto& [incoming_node_name, incoming_message] :
                 this->incoming_messages_per_time_slice.at(time_step)
                     .node_name_to_messages) {

                if (marginal.rows() == 0) {
                    marginal = incoming_message(0, 0);
                }
                else {
                    marginal =
                        marginal.array() * incoming_message(0, 0).array();
                }
            }

            if (normalized) {
                // Outliers can result in zero vector probabilities. Adding
                // a noise to generate a uniform distribution after
                // normalization.
                marginal = marginal.array() + EPSILON;
                Eigen::VectorXd sum_per_row = marginal.rowwise().sum().array();
                marginal =
                    (marginal.array().colwise() / sum_per_row.array()).matrix();

                if (aggregation_value > 0) {
                    Eigen::VectorXd fixed_probs =
                        1 - marginal.col(aggregation_value).array();
                    Eigen::MatrixXd new_marginal =
                        marginal.col(aggregation_value).array() /
                        (marginal.cols() - 1);
                    new_marginal = new_marginal.replicate(1, marginal.cols());
                    new_marginal.col(aggregation_value) = fixed_probs;
                }
            }

            return marginal;
        }

        void VariableNode::set_data_at(int time_step,
                                       const Eigen::MatrixXd& data) {

            // Convert each element of the vector to a binary row vector and
            // stack them horizontally;
            Eigen::MatrixXd data_matrix(data.size(), this->cardinality);
            for (int i = 0; i < data.size(); i++) {
                Eigen::VectorXd binary_vector =
                    Eigen::VectorXd::Zero(this->cardinality);
                binary_vector[data(i, 0)] = 1;
                data_matrix.row(i) = move(binary_vector);
            }

            this->data_per_time_slice[time_step] = data_matrix;
        }

        void VariableNode::erase_data_at(int time_step) {
            this->data_per_time_slice.erase(time_step);
        }

        void VariableNode::add_backward_blocking(
            const std::string& incoming_node_label,
            const std::string& target_node_label) {
            this->backward_blocking[target_node_label].insert(
                incoming_node_label);
        }

        //----------------------------------------------------------------------
        // Getters & Setters
        //----------------------------------------------------------------------
        int VariableNode::get_cardinality() const { return cardinality; }

    } // namespace model
} // namespace tomcat
