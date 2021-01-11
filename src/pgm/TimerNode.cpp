#include "TimerNode.h"

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        TimerNode::TimerNode(const shared_ptr<NodeMetadata>& metadata,
                             int time_step)
            : RandomVariableNode(metadata, time_step) {}

        TimerNode::TimerNode(shared_ptr<NodeMetadata>&& metadata, int time_step)
            : RandomVariableNode(move(metadata), time_step) {}

        TimerNode::~TimerNode() {}

        //----------------------------------------------------------------------
        // Copy & Move constructors/assignments
        //----------------------------------------------------------------------
        TimerNode::TimerNode(const TimerNode& node)
            : RandomVariableNode(node.metadata, node.time_step) {
            this->copy_node(node);
        }

        TimerNode& TimerNode::operator=(const TimerNode& node) {
            this->copy_node(node);
            return *this;
        }

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        void TimerNode::copy_node(const TimerNode& node) {
            RandomVariableNode::copy_node(node);
            this->backward_assignment = node.backward_assignment;
        }

        unique_ptr<Node> TimerNode::clone() const {
            unique_ptr<TimerNode> new_node = make_unique<TimerNode>(*this);
            new_node->metadata = make_shared<NodeMetadata>(*this->metadata);
            new_node->clone_cpd_templates();
            new_node->clone_cpd();
            return new_node;
        }

        // ---------------------------------------------------------------------
        // Getters & Setters
        // ---------------------------------------------------------------------
        const Eigen::MatrixXd& TimerNode::get_backward_assignment() const {
            return backward_assignment;
        }

        void TimerNode::set_backward_assignment(
            const Eigen::MatrixXd& backward_assignment) {
            this->backward_assignment = backward_assignment;
        }

    } // namespace model
} // namespace tomcat
