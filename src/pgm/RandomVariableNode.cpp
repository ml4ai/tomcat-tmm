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
            this->timer = node.timer;
            this->children = node.children;
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
            // The following attributes have to be set again according to the
            // connections of this new copy in a DBN
            new_node->cpd = nullptr;
            new_node->parents.clear();
            new_node->timer = nullptr;
            new_node->timed_copies = nullptr;
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

        void RandomVariableNode::update_cpd_templates_dependencies(
            const NodeMap& parameter_nodes_map) {
            for (auto& mapping : this->cpd_templates) {
                mapping.second->update_dependencies(parameter_nodes_map);
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
                sample = this->sample_from_conjugacy(
                    random_generator_per_job.at(0), this->get_size());
            }
            else {
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
                if (child == this->timer) {
                    // The timer that controls this node will have posterior
                    // weights processed next, ut segment posteriors.
                    continue;
                }

                shared_ptr<RandomVariableNode> rv_child =
                    dynamic_pointer_cast<RandomVariableNode>(child);

                Eigen::MatrixXd child_weights =
                    rv_child->get_cpd()->get_posterior_weights(
                        rv_child->get_parents(),
                        shared_from_this(),
                        rv_child,
                        num_jobs);

                Eigen::MatrixXd child_log_weights =
                    (child_weights.array() + EPSILON).log();

                if (!this->get_metadata()->is_in_plate()) {
                    // Multiply the weights of each one of the assignments of
                    // the child node. When an off-plate node is
                    // parent of an in-plate node, all of the in-plate
                    // instances of the child node are also considered
                    // children of the off-plate node, and, therefore, their
                    // weights must be multiplied together.
                    child_log_weights = child_log_weights.rowwise().sum();
                }

                log_weights = (log_weights.array() + child_log_weights.array());
            }

            // Include posterior weights of immediate segments for nodes
            // that are time controlled.
            Eigen::MatrixXd segment_log_weights =
                this->get_segments_log_posterior_weights(num_jobs);
            if (segment_log_weights.size() > 0) {
                log_weights =
                    (log_weights.array() + segment_log_weights.array());
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

            if (!this->has_timer()) {
                // No weights if the node is not controlled by a timer
                return segments_weights;
            }

            const auto& left_state = this->get_previous();
            const auto& right_state = this->get_next();

            const auto& central_timer = this->get_timer();
            const auto& left_last_timer =
                left_state ? left_state->get_timer() : nullptr;
            const auto& right_first_timer =
                right_state ? right_state->get_timer() : nullptr;

            // Last time step being sampled. This will be used to deal with
            // right segment truncation in the computation of the segment
            // posteriors.
            int last_time_step = this->timed_copies->size() - 1;

            // Left segment
            if (left_state) {
                // Left segment
                Eigen::MatrixXd left_seg_weights =
                    left_last_timer->get_cpd()
                        ->get_left_segment_posterior_weights(
                            left_last_timer,
                            right_state,
                            this->get_time_step(),
                            last_time_step,
                            num_jobs);
                segments_weights = (left_seg_weights.array() + EPSILON).log();
            }

            // Central segment
            Eigen::MatrixXd central_seg_weights =
                this->timer->get_cpd()->get_central_segment_posterior_weights(
                    left_state,
                    central_timer,
                    right_state,
                    last_time_step,
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
            if (right_state) {
                Eigen::MatrixXd right_seg_weights =
                    right_first_timer->get_cpd()
                        ->get_right_segment_posterior_weights(
                            right_first_timer, last_time_step, num_jobs);

                segments_weights =
                    (segments_weights.array() +
                     (right_seg_weights.array() + EPSILON).log());
            }

            return segments_weights;
        }

        Eigen::MatrixXd RandomVariableNode::sample_from_conjugacy(
            const shared_ptr<gsl_rng>& random_generator,
            int num_samples) const {
            return this->cpd->sample_from_conjugacy(
                random_generator, num_samples, shared_from_this());
        }

        Eigen::VectorXd RandomVariableNode::get_pdfs(int num_jobs,
                                                     int parameter_idx) const {
            return this->cpd->get_pdfs(
                shared_from_this(), num_jobs, parameter_idx);
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

            if (this->timed_copies) {
                int t0 = this->get_metadata()->get_initial_time_step();
                if (this->time_step - t0 + increment <
                    this->timed_copies->size()) {
                    next =
                        (*this->timed_copies)[this->time_step - t0 + increment];
                }
            }

            return next;
        }

        bool RandomVariableNode::is_continuous() const {
            return this->cpd->is_continuous();
        }

        void RandomVariableNode::print_cpds(std::ostream& output_stream) const {
            for (const auto& [id, cpd] : this->cpd_templates) {
                output_stream << this->get_metadata()->get_label() << " | ";
                cpd->print(output_stream);
                output_stream << endl;
            }
        }

        bool RandomVariableNode::is_segment_dependency() const {
            for (const auto& child : this->children) {
                if (dynamic_pointer_cast<RandomVariableNode>(child)
                        ->has_timer() ||
                    child->get_metadata()->is_timer()) {
                    return true;
                }
            }

            return false;
        }

        const RVNodePtrVec&
        RandomVariableNode::get_children(int time_step) const {
            return this->children_per_time_step.at(time_step);
        }

        void RandomVariableNode::set_assignment(int i, int j, double value) {
            if (!this->frozen) {
                this->assignment(i, j) = value;
            }
        }

        bool RandomVariableNode::has_child_at(int time_step) const {
            return this->children_per_time_step.size() > time_step &&
                   !this->children_per_time_step.at(time_step).empty();
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

        void RandomVariableNode::set_children(
            const vector<shared_ptr<Node>>& children) {
            this->children = children;

            for (const auto& child : children) {
                auto rv_child = dynamic_pointer_cast<RandomVariableNode>(child);

                int new_size = max(rv_child->get_time_step() + 1,
                                   (int)this->children_per_time_step.size());
                this->children_per_time_step.resize(new_size);
                this->children_per_time_step[rv_child->get_time_step()]
                    .push_back(rv_child);

                if (child->get_metadata()->is_timer()) {
                    this->child_timer = true;
                }

                if (!child->get_metadata()->is_replicable()) {
                    this->single_time_children.push_back(rv_child);
                }
            }
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

        bool RandomVariableNode::has_child_timer() const {
            return this->child_timer;
        }

        const RVNodePtrVec&
        RandomVariableNode::get_single_time_children() const {
            return single_time_children;
        }

    } // namespace model
} // namespace tomcat
