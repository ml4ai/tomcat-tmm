#include "MarginalizationFactorNode.h"

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        MarginalizationFactorNode::MarginalizationFactorNode(
            const string& label,
            int time_step,
            const CPD::TableOrderingMap& joint_ordering_map,
            const string& joint_node_label)
            : FactorNode(compose_label(label),
                         time_step,
                         Eigen::MatrixXd::Zero(0, 0),
                         joint_ordering_map,
                         ""),
              joint_node_label(joint_node_label) {

            this->joint_cardinality = 1;
            for (const auto& [node_label, indexing_scheme] :
                 joint_ordering_map) {
                this->joint_cardinality *= indexing_scheme.cardinality;
            }
        }

        MarginalizationFactorNode::~MarginalizationFactorNode() {}

        //----------------------------------------------------------------------
        // Copy & Move constructors/assignments
        //----------------------------------------------------------------------
        MarginalizationFactorNode::MarginalizationFactorNode(
            const MarginalizationFactorNode& factor_node) {
            this->copy_node(factor_node);
        }

        MarginalizationFactorNode& MarginalizationFactorNode::operator=(
            const MarginalizationFactorNode& factor_node) {
            this->copy_node(factor_node);
            return *this;
        }

        //----------------------------------------------------------------------
        // Static functions
        //----------------------------------------------------------------------
        std::string MarginalizationFactorNode::compose_label(
            const std::string& original_label) {
            return "m(" + original_label + ")";
        }

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        void MarginalizationFactorNode::copy_node(
            const MarginalizationFactorNode& node) {
            FactorNode::copy_node(node);
            this->joint_node_label = node.joint_node_label;
            this->joint_cardinality = node.joint_cardinality;
        }

        bool MarginalizationFactorNode::is_segment() const { return true; }

        Tensor3 MarginalizationFactorNode::get_outward_message_to(
            const shared_ptr<MessageNode>& template_target_node,
            int template_time_step,
            int target_time_step,
            Direction direction) const {

            Tensor3 output_message;

            if (direction == Direction::forward) {
                if (target_time_step == 0) {
                    // In the first time step, the joint distribution is
                    // given by the product of the dependencies of this
                    // factor node. Later, the joint probability will be
                    // computed from the segment distributions.
                    vector<Tensor3> messages =
                        this->get_incoming_messages_in_order(
                            template_target_node->get_label(),
                            time_step,
                            target_time_step,
                            this->original_potential.potential);
                    output_message = get_cartesian_tensor(messages);
                }
                else {
                    // At this point, the joint distribution already contains
                    // the full distribution. The nodes in the joint
                    // distribution are not independent anymore after the
                    // time step 0, they will depend on each other as
                    // segments are expanded or closed.
                    int num_data_points;
                    for (const auto& [node_name, incoming_message] :
                         this->incoming_messages_per_time_slice
                             .at(template_time_step)
                             .node_name_to_messages) {
                        num_data_points = incoming_message.get_shape()[1];
                        break;
                    }
                    output_message = Tensor3::ones(
                        1, num_data_points, this->joint_cardinality);
                }
            }
            else {
                // Marginalize other nodes out
                Tensor3 joint_message =
                    this->incoming_messages_per_time_slice
                        .at(template_time_step)
                        .get_message_for(this->joint_node_label,
                                         template_time_step);

                const auto& ordering_map =
                    this->original_potential.potential.ordering_map.at(
                        template_target_node->get_label());

                // We will place the probability for each one of the values
                // of the target node in the columns of a tensor. Each row
                // represents a combination of assignments of each node in
                // the joint distribution.
                int num_rows =
                    joint_message.get_shape()[2] / ordering_map.cardinality;
                int num_cols = ordering_map.cardinality;

                Eigen::MatrixXd output_matrix = Eigen::MatrixXd(
                    joint_message.get_shape()[1], ordering_map.cardinality);

                for (int row = 0; row < joint_message.get_shape()[1]; row++) {
                    Eigen::VectorXd joint = Eigen::VectorXd ::Zero(num_cols);
                    for (int col = 0; col < joint_message.get_shape()[2];
                         col++) {
                        int j = col / ordering_map.right_cumulative_cardinality;
                        j = j % ordering_map.cardinality;

                        joint(j) = joint(j) + joint_message(0, row, col);
                    }
                    output_matrix.row(row) = joint;
                }
                output_message = Tensor3(output_matrix);
            }

            return output_message;
        }

        //----------------------------------------------------------------------
        // Getters & Setters
        //----------------------------------------------------------------------

    } // namespace model
} // namespace tomcat
