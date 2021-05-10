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
        std::string VariableNode::compose_segment_label(
            const std::string& timed_node_label) {
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
                MessageContainer message_container =
                    this->incoming_messages_per_time_slice.at(
                        template_time_step);

                for (const auto& [incoming_node_name, incoming_message] :
                     message_container.node_name_to_messages) {

                    if (incoming_node_name ==
                        MessageNode::get_name(template_target_node->get_label(),
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

                if(!this->is_segment() || template_target_node->is_segment()) {
                    // We do not marginalize if a segment is sending a
                    // message to a marginalization factor, otherwise we
                    // would lose the contribution of the dependencies of the
                    // segment duration distribution
                    outward_message.normalize_rows();
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
            }

            return marginal;
        }

        void VariableNode::set_data_at(int time_step,
                                       const Eigen::VectorXd& data,
                                       bool aggregate) {

            int cols = aggregate ? 2 : this->cardinality;
            // Convert each element of the vector to a binary row vector and
            // stack them horizontally;
            Eigen::MatrixXd data_matrix(data.size(), cols);
            for (int i = 0; i < data.size(); i++) {
                Eigen::VectorXd binary_vector = Eigen::VectorXd::Zero(cols);
                binary_vector[data[i]] = 1;
                data_matrix.row(i) = move(binary_vector);
            }

            this->data_per_time_slice[time_step] = data_matrix;
        }

        void VariableNode::erase_data_at(int time_step) {
            this->data_per_time_slice.erase(time_step);
        }

        //----------------------------------------------------------------------
        // Getters & Setters
        //----------------------------------------------------------------------
        int VariableNode::get_cardinality() const { return cardinality; }

    } // namespace model
} // namespace tomcat
