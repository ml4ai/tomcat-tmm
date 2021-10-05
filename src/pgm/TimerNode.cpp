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
            this->forward_assignment = node.forward_assignment;
        }

        unique_ptr<Node> TimerNode::clone() const {
            unique_ptr<TimerNode> new_node = make_unique<TimerNode>(*this);
            new_node->metadata = make_shared<NodeMetadata>(*this->metadata);
            new_node->clone_cpd_templates();
            new_node->clone_cpd();
            return new_node;
        }

        Eigen::MatrixXd TimerNode::sample_from_posterior(
            const vector<shared_ptr<gsl_rng>>& random_generator_per_job,
            const std::vector<int>& time_steps_per_sample) {
            // Timer is not sampled in parallel since it only requires
            // straight forward matrix operations.

            int data_size = this->controlled_node->get_size();
            Eigen::MatrixXd sample = Eigen::MatrixXd::Zero(data_size, 1);
            if (const auto& previous_timer =
                    dynamic_pointer_cast<TimerNode>(this->get_previous())) {
                const auto& previous_controlled_node =
                    this->controlled_node->get_previous();

                const Eigen::VectorXi& previous_values =
                    previous_controlled_node->get_assignment()
                        .col(0)
                        .cast<int>();
                const Eigen::VectorXi& current_values =
                    this->controlled_node->get_assignment().col(0).cast<int>();

                Eigen::VectorXi equal =
                    (previous_values.array() == current_values.array())
                        .cast<int>();

                const Eigen::VectorXi& previous_durations =
                    previous_timer->get_forward_assignment().col(0).cast<int>();
                sample.col(0) =
                    ((previous_durations.array() + 1) * equal.array())
                        .cast<double>();
            }

            return sample;
        }

        void TimerNode::update_backward_assignment() {
            const auto& next_timer =
                dynamic_pointer_cast<TimerNode>(this->get_next());

            int rows = this->controlled_node->get_size();
            if (next_timer) {
                const auto& next_controlled_node =
                    this->controlled_node->get_next();

                this->assignment = Eigen::MatrixXd(rows, 1);
                for (int i = 0; i < rows; i++) {
                    if (next_controlled_node->get_assignment()(i, 0) ==
                        this->controlled_node->get_assignment()(i, 0)) {
                        // Controlled node does not transition to a different
                        // state, therefore, the segment is the same. We just
                        // increment the timer to the new segment size.
                        this->assignment.row(i) =
                            next_timer->get_backward_assignment()
                                .row(i)
                                .array() +
                            1;
                    }
                    else {
                        // Beginning of a new segment
                        this->assignment(i, 0) = 0;
                    }
                }
            }
            else {
                this->assignment = Eigen::MatrixXd::Zero(rows, 1);
            }
        }

        Eigen::VectorXd TimerNode::get_left_segment_posterior_weights(
            int left_segment_duration,
            const shared_ptr<RandomVariableNode>& right_segment_state,
            int central_segment_time_step,
            int last_time_step,
            int sample_idx) const {

            return this->get_cpd()->get_single_left_segment_posterior_weights(
                dynamic_pointer_cast<const TimerNode>(shared_from_this()),
                left_segment_duration,
                right_segment_state,
                central_segment_time_step,
                last_time_step,
                sample_idx);
        }

        // ---------------------------------------------------------------------
        // Getters & Setters
        // ---------------------------------------------------------------------
        const Eigen::MatrixXd& TimerNode::get_forward_assignment() const {
            return forward_assignment;
        }

        Eigen::MatrixXd TimerNode::get_forward_assignment(
            const ProcessingBlock& processing_block) const {
            return this->forward_assignment.block(
                processing_block.first,
                0,
                processing_block.second,
                this->forward_assignment.cols());
        }

        double TimerNode::get_forward_assignment(int i, int j) const {
            return this->forward_assignment(i, j);
        }

        void TimerNode::set_forward_assignment(
            const Eigen::MatrixXd& forward_assignment) {
            this->forward_assignment = forward_assignment;
        }

        void TimerNode::set_forward_assignment(
            const Eigen::MatrixXd& forward_assignment,
            const ProcessingBlock& processing_block) {
            this->forward_assignment.block(processing_block.first,
                                           0,
                                           processing_block.second,
                                           this->forward_assignment.cols()) =
                forward_assignment;
        }

        const Eigen::MatrixXd& TimerNode::get_backward_assignment() const {
            return assignment;
        }

        Eigen::MatrixXd TimerNode::get_backward_assignment(
            const ProcessingBlock& processing_block) const {
            return Node::get_assignment(processing_block);
        }

        double TimerNode::get_backward_assignment(int i, int j) const {
            return Node::get_assignment(i, j);
        }

        void TimerNode::set_backward_assignment(
            const Eigen::MatrixXd& backward_assignment) {
            this->assignment = backward_assignment;
        }

        void TimerNode::set_backward_assignment(
            const Eigen::MatrixXd& backward_assignment,
            const ProcessingBlock& processing_block) {
            Node::set_assignment(backward_assignment, processing_block);
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
