#include "RandomVariableNode.h"

#include <algorithm>
#include <fmt/format.h>
#include <iterator>
#include <stdexcept>

#include "TimerNode.h"

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        RandomVariableNode::RandomVariableNode(
            const shared_ptr<NodeMetadata>& metadata, int time_step)
            : Node(metadata), time_step(time_step) {}

        RandomVariableNode::RandomVariableNode(
            shared_ptr<NodeMetadata>&& metadata, int time_step)
            : Node(move(metadata)), time_step(time_step) {}

        RandomVariableNode::~RandomVariableNode() {}

        //----------------------------------------------------------------------
        // Copy & Move constructors/assignments
        //----------------------------------------------------------------------
        RandomVariableNode::RandomVariableNode(const RandomVariableNode& node) {
            this->copy_node(node);
        }

        RandomVariableNode&
        RandomVariableNode::operator=(const RandomVariableNode& node) {
            this->copy_node(node);
            return *this;
        }

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        void RandomVariableNode::copy_node(const RandomVariableNode& node) {
            this->metadata = node.metadata;
            this->time_step = node.time_step;
            this->assignment = node.assignment;
            this->cpd_templates = node.cpd_templates;
            this->cpd = node.cpd;
            this->frozen = node.frozen;
            this->parents = node.parents;
            this->children = node.children;
            this->timer = node.timer;
            this->timed_copies = node.timed_copies;
        }

        string RandomVariableNode::get_description() const {

            if (this->assignment.size() == 1) {
                stringstream assignment_string;
                assignment_string << this->assignment;

                return fmt::format("RV({}, {}, {})",
                                   this->metadata->get_label(),
                                   this->time_step,
                                   assignment_string.str());
            }
            else {
                stringstream assignment_string;
                assignment_string << this->assignment.transpose();

                return fmt::format("RV({}, {}, [{}])",
                                   this->metadata->get_label(),
                                   this->time_step,
                                   assignment_string.str());
            }
        }

        unique_ptr<Node> RandomVariableNode::clone() const {
            unique_ptr<RandomVariableNode> new_node =
                make_unique<RandomVariableNode>(*this);
            new_node->metadata = make_shared<NodeMetadata>(*this->metadata);
            new_node->clone_cpd_templates();
            new_node->clone_cpd();
            return new_node;
        }

        void RandomVariableNode::clone_cpd_templates() {
            for (auto& mapping : this->cpd_templates) {
                mapping.second = mapping.second->clone();
            }
        }

        void RandomVariableNode::clone_cpd() {
            if (this->cpd != nullptr) {
                this->cpd = this->cpd->clone();
            }
        }

        string RandomVariableNode::get_timed_name() const {
            return this->metadata->get_timed_name(this->time_step);
        }

        void RandomVariableNode::reset_cpd_updated_status() {
            this->cpd->reset_updated_status();
        }

        void RandomVariableNode::update_cpd_templates_dependencies(
            const NodeMap& parameter_nodes_map, int time_step) {
            for (auto& mapping : this->cpd_templates) {
                if (!mapping.second->is_updated()) {
                    mapping.second->update_dependencies(parameter_nodes_map,
                                                        time_step);
                }
            }
        }

        Eigen::MatrixXd RandomVariableNode::sample(
            const vector<shared_ptr<gsl_rng>>& random_generator_per_job,
            int num_samples) const {

            return this->cpd->sample(
                random_generator_per_job, num_samples, shared_from_this());
        }

        Eigen::MatrixXd RandomVariableNode::sample_from_posterior(
            const vector<shared_ptr<gsl_rng>>& random_generator_per_job) {
            Eigen::MatrixXd sample;

            if (this->metadata->is_parameter()) {
                sample = this->sample_from_conjugacy
                    (random_generator_per_job.at(0), this->get_size());
            } else {
                int num_jobs = random_generator_per_job.size();
                Eigen::MatrixXd weights = this->get_posterior_weights(num_jobs);
                sample = this->cpd->sample_from_posterior(
                    random_generator_per_job, weights, shared_from_this());
            }

            return sample;
        }

        Eigen::MatrixXd
        RandomVariableNode::get_posterior_weights(int num_jobs) {
            int rows = this->get_size();
            int cols = this->get_metadata()->get_cardinality();
            Eigen::MatrixXd log_weights = Eigen::MatrixXd::Zero(rows, cols);

            for (auto& child : this->children) {
                shared_ptr<RandomVariableNode> rv_child =
                    dynamic_pointer_cast<RandomVariableNode>(child);

                Eigen::MatrixXd child_weights =
                    rv_child->get_cpd()->get_posterior_weights(
                        rv_child->get_parents(),
                        shared_from_this(),
                        rv_child,
                        num_jobs);
                if (this->get_metadata()->is_in_plate()) {
                    // Multiply weights of each one of the assignments
                    // separately.
                    log_weights = (log_weights.array() +
                                   (child_weights.array() + EPSILON).log());
                }
                else {
                    // Multiply the weights of each one of the assignments of
                    // the child node. When an off-plate node is
                    // parent of an in-plate node, all of the in-plate
                    // instances of the child node are also considered
                    // children of the off-plate node, and, therefore, their
                    // weights must be multiplied together.
                    Eigen::MatrixXd agg_child_weights =
                        (child_weights.array() + EPSILON).log().rowwise().sum();
                    log_weights =
                        (log_weights.array() + agg_child_weights.array());
                }
            }

            // Include posterior weights of immediate segments for nodes that
            // are time controlled.
            Eigen::MatrixXd segments_weights =
                this->get_segments_log_posterior_weights(num_jobs);
            if (segments_weights.size() > 0) {
                log_weights = (log_weights.array() + segments_weights.array());
            }

            // Unlog and normalize the weights
            log_weights.colwise() -= log_weights.rowwise().maxCoeff();
            log_weights = log_weights.array().exp();
            Eigen::VectorXd sum_per_row = log_weights.rowwise().sum();
            return (log_weights.array().colwise() / sum_per_row.array());
        }

        Eigen::MatrixXd
        RandomVariableNode::get_segments_log_posterior_weights(int num_jobs) {
            Eigen::MatrixXd segments_weights(0, 0);

            if (this->has_timer()) {
                // Left segment
                if (const auto& previous_timer =
                        dynamic_pointer_cast<TimerNode>(
                            this->timer->get_previous())) {
                    // Left segment
                    Eigen::MatrixXd left_seg_weights =
                        previous_timer->get_cpd()
                            ->get_left_segment_posterior_weights(previous_timer,
                                                                 num_jobs);
                    segments_weights =
                        (left_seg_weights.array() + EPSILON).log();
                }

                // Central segment
                Eigen::MatrixXd central_seg_weights =
                    this->timer->get_cpd()
                        ->get_central_segment_posterior_weights(this->timer,
                                                                num_jobs);
                if (segments_weights.size() > 0) {
                    segments_weights =
                        (segments_weights.array() +
                         (central_seg_weights.array() + EPSILON).log());
                }
                else {
                    segments_weights =
                        (central_seg_weights.array() + EPSILON).log();
                }

                // Right segment
                if (const auto& next_timer = dynamic_pointer_cast<TimerNode>(
                        this->timer->get_next())) {

                    Eigen::MatrixXd right_seg_weights =
                        next_timer->get_cpd()
                            ->get_right_segment_posterior_weights(next_timer,
                                                                  num_jobs);

                    segments_weights =
                        (segments_weights.array() +
                         (right_seg_weights.array() + EPSILON).log());
                }
            }

            return segments_weights;
        }

        Eigen::MatrixXd RandomVariableNode::sample_from_conjugacy(
            const shared_ptr<gsl_rng>& random_generator,
            int num_samples) const {
            return this->cpd->sample_from_conjugacy(
                random_generator, num_samples, shared_from_this());
        }

        void RandomVariableNode::update_parents_sufficient_statistics() {
            this->cpd->update_sufficient_statistics(shared_from_this());
        }

        void RandomVariableNode::add_to_sufficient_statistics(
            const vector<double>& values) {
            this->cpd->add_to_sufficient_statistics(values);
        }

        void RandomVariableNode::reset_sufficient_statistics() {
            this->cpd->reset_sufficient_statistics();
        }

        void RandomVariableNode::freeze() { RandomVariableNode::frozen = true; }

        void RandomVariableNode::unfreeze() {
            RandomVariableNode::frozen = false;
        }

        void RandomVariableNode::add_cpd_template(const shared_ptr<CPD>& cpd) {
            this->check_cpd(cpd);
            this->cpd_templates[cpd->get_id()] = cpd;
        }

        void RandomVariableNode::add_cpd_template(shared_ptr<CPD>&& cpd) {
            this->check_cpd(cpd);
            this->cpd_templates[cpd->get_id()] = move(cpd);
        }

        void RandomVariableNode::check_cpd(const shared_ptr<CPD>& cpd) const {
            const auto& indexing_mapping = cpd->get_parent_label_to_indexing();
            const auto& label = this->metadata->get_label();

            if (EXISTS(label, indexing_mapping)) {
                if (indexing_mapping.at(label).order != 0) {
                    throw TomcatModelException("Replicable nodes that depend "
                                               "on itself over time must be "
                                               "the first node to index a CPD"
                                               ".");
                }
            }
        }

        shared_ptr<CPD> RandomVariableNode::get_cpd_for(
            const vector<string>& parent_labels) const {
            string key = this->get_unique_key_from_labels(parent_labels);
            shared_ptr<CPD> cpd;
            if (EXISTS(key, this->cpd_templates)) {
                cpd = this->cpd_templates.at(key);
            }
            else {
                throw invalid_argument(
                    "No CPD found associated with the parents informed.");
            }

            return cpd;
        }

        string RandomVariableNode::get_unique_key_from_labels(
            const vector<string>& labels) const {

            vector<string> mutable_labels = labels;
            stringstream ss;
            sort(mutable_labels.begin(), mutable_labels.end());
            copy(mutable_labels.begin(),
                 mutable_labels.end(),
                 ostream_iterator<string>(ss, ","));
            return ss.str();
        }

        bool RandomVariableNode::has_timer() const {
            return this->timer != nullptr;
        }

        shared_ptr<RandomVariableNode>
        RandomVariableNode::get_previous(int increment) const {
            shared_ptr<RandomVariableNode> previous;

            int t0 = this->get_metadata()->get_initial_time_step();
            if (this->time_step - t0 - increment >= 0) {
                previous =
                    (*this->timed_copies)[this->time_step - t0 - increment];
            }

            return previous;
        }

        shared_ptr<RandomVariableNode>
        RandomVariableNode::get_next(int increment) const {
            shared_ptr<RandomVariableNode> next;

            int t0 = this->get_metadata()->get_initial_time_step();
            if (this->time_step - t0 + increment < this->timed_copies->size()) {
                next = (*this->timed_copies)[this->time_step - t0 + increment];
            }

            return next;
        }

        // ---------------------------------------------------------------------
        // Getters & Setters
        // ---------------------------------------------------------------------
        int RandomVariableNode::get_time_step() const { return time_step; }

        void RandomVariableNode::set_time_step(int time_step) {
            this->time_step = time_step;
        }

        void
        RandomVariableNode::set_assignment(const Eigen::MatrixXd& assignment) {
            if (!this->frozen) {
                this->assignment = assignment;
            }
        }

        bool RandomVariableNode::is_frozen() const { return frozen; }

        void RandomVariableNode::set_cpd(const shared_ptr<CPD>& cpd) {
            this->cpd = cpd;
        }
        const shared_ptr<CPD>& RandomVariableNode::get_cpd() const {
            return cpd;
        }

        const vector<shared_ptr<Node>>&
        RandomVariableNode::get_parents() const {
            return parents;
        }

        void RandomVariableNode::set_parents(
            const vector<shared_ptr<Node>>& parents) {
            this->parents = parents;
        }

        const vector<shared_ptr<Node>>&
        RandomVariableNode::get_children() const {
            return children;
        }

        void RandomVariableNode::set_children(
            const vector<shared_ptr<Node>>& children) {
            this->children = children;
        }

        const shared_ptr<TimerNode>& RandomVariableNode::get_timer() const {
            return timer;
        }

        void RandomVariableNode::set_timer(const shared_ptr<TimerNode>& timer) {
            this->timer = timer;
        }

        void RandomVariableNode::set_timed_copies(
            const shared_ptr<vector<shared_ptr<RandomVariableNode>>>&
                timed_copies) {
            this->timed_copies = timed_copies;
        }

    } // namespace model
} // namespace tomcat
