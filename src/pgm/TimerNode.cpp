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

        Eigen::MatrixXd TimerNode::sample_from_posterior(
            const std::shared_ptr<gsl_rng>& random_generator) {

            int rows = this->controlled_node->get_size();
            if (const auto& previous_timer = this->get_previous()) {
                const auto& previous_controlled_node =
                    this->controlled_node->get_previous();

                Eigen::MatrixXd sample(rows, 1);
                for (int i = 0; i < rows; i++) {
                    if (previous_controlled_node->get_assignment()(i, 0) ==
                        this->controlled_node->get_assignment()(i, 0)) {
                        // Controlled node does not transition to a different
                        // state, therefore, the segment is the same. We just
                        // increment the timer to the new segment size.
                        sample.row(i) =
                            previous_timer->get_assignment().row(i).array() + 1;
                    }
                    else {
                        // Beginning of a new segment
                        sample(i, 0) = 0;
                    }
                }

                return sample;
            }
            else {
                return Eigen::MatrixXd::Zero(rows, 1);
            }
        }

        void TimerNode::update_backward_assignment() {
            int rows = this->controlled_node->get_size();
            if (const auto& next_timer = this->get_timer()) {
                const auto& next_controlled_node =
                    this->controlled_node->get_next();

                this->backward_assignment = Eigen::MatrixXd(rows, 1);
                for (int i = 0; i < rows; i++) {
                    if (next_controlled_node->get_assignment()(i, 0) ==
                        this->controlled_node->get_assignment()(i, 0)) {
                        // Controlled node does not transition to a different
                        // state, therefore, the segment is the same. We just
                        // increment the timer to the new segment size.
                        this->backward_assignment.row(i) =
                            next_timer->get_backward_assignment()
                                .row(i)
                                .array() +
                            1;
                    }
                    else {
                        // Beginning of a new segment
                        this->backward_assignment(i, 0) = 0;
                    }
                }
            }
            else {
                this->backward_assignment = Eigen::MatrixXd::Zero(rows, 1);
            }
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

        const shared_ptr<RandomVariableNode>&
        TimerNode::get_controlled_node() const {
            return controlled_node;
        }

        void TimerNode::set_controlled_node(
            const shared_ptr<RandomVariableNode>& controlled_node) {
            this->controlled_node = controlled_node;
        }

    } // namespace model
} // namespace tomcat
