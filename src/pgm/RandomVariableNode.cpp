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
            this->timed_copies = node.timed_copies;
            this->cached_posterior_weights = node.cached_posterior_weights;
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
            const vector<shared_ptr<gsl_rng>>& random_generator_per_job,
            int min_time_step_to_sample,
            int max_time_step_to_sample,
            bool use_weights_cache) {
            Eigen::MatrixXd sample;

            if (this->metadata->is_parameter()) {
                sample = this->sample_from_conjugacy(
                    random_generator_per_job.at(0), this->get_size());
            }
            else {
                int num_jobs = random_generator_per_job.size();
                Eigen::MatrixXd weights =
                    this->get_posterior_weights(num_jobs,
                                                min_time_step_to_sample,
                                                max_time_step_to_sample,
                                                use_weights_cache);
                sample = this->cpd->sample_from_posterior(
                    random_generator_per_job, weights, shared_from_this());
            }

            return sample;
        }

        Eigen::MatrixXd
        RandomVariableNode::get_posterior_weights(int num_jobs,
                                                  int min_time_step_to_sample,
                                                  int max_time_step_to_sample,
                                                  bool use_weights_cache) {

            Eigen::MatrixXd log_weights(0, 0);
            if (use_weights_cache) {
                this->accumulate_cached_log_weights(min_time_step_to_sample);
                log_weights =
                    this->get_cached_log_weights(min_time_step_to_sample);
            }

            if (log_weights.size() == 0) {
                int rows = this->get_size();
                int cols = this->get_metadata()->get_cardinality();
                log_weights = Eigen::MatrixXd::Zero(rows, cols);
            }

            for (auto& child : this->get_children_in_range(
                     min_time_step_to_sample, max_time_step_to_sample)) {

                shared_ptr<RandomVariableNode> rv_child =
                    dynamic_pointer_cast<RandomVariableNode>(child);

                if (rv_child->get_time_step() > max_time_step_to_sample) {
                    continue;
                }

                Eigen::MatrixXd child_weights =
                    rv_child->get_cpd()->get_posterior_weights(
                        rv_child->get_parents(),
                        shared_from_this(),
                        rv_child,
                        num_jobs);
                Eigen::MatrixXd child_log_weights(0, 0);

                if (this->get_metadata()->is_in_plate()) {
                    // Multiply weights of each one of the assignments
                    // separately.
                    child_log_weights = (child_weights.array() + EPSILON).log();
                }
                else {
                    // Multiply the weights of each one of the assignments of
                    // the child node. When an off-plate node is
                    // parent of an in-plate node, all of the in-plate
                    // instances of the child node are also considered
                    // children of the off-plate node, and, therefore, their
                    // weights must be multiplied together.
                    child_log_weights =
                        (child_weights.array() + EPSILON).log().rowwise().sum();
                }

                if (use_weights_cache) {
                    this->cache_current_log_weights(
                        min_time_step_to_sample, rv_child, child_log_weights);
                }

                log_weights = (log_weights.array() + child_log_weights.array());
            }

            // Include posterior weights of immediate segments for nodes that
            // are time controlled.
            Eigen::MatrixXd segments_weights =
                this->get_segments_log_posterior_weights(
                    num_jobs, max_time_step_to_sample);
            if (segments_weights.size() > 0) {
                log_weights = (log_weights.array() + segments_weights.array());
            }

            // Unlog and normalize the weights
            log_weights.colwise() -= log_weights.rowwise().maxCoeff();
            log_weights = log_weights.array().exp();
            Eigen::VectorXd sum_per_row = log_weights.rowwise().sum();
            return (log_weights.array().colwise() / sum_per_row.array());
        }

        void RandomVariableNode::accumulate_cached_log_weights(
            int min_time_step_to_sample) {

            if (this->metadata->is_multitime()) {
                // Multi-time nodes can have weights computed previously
                // cached when we perform forward inference. Other kinds of
                // nodes do not have connections with previously processed
                // nodes, therefore, they don't have any posterior weight to
                // be cached.
                if(min_time_step_to_sample < this->cached_posterior_weights.time_step) {
                    this->clear_cache();
                    return;
                }

                if (min_time_step_to_sample >
                    this->cached_posterior_weights.time_step + 1) {

                    auto& cum_log_weights = this->cached_posterior_weights
                                                .cum_log_weights_from_children;
                    auto& prev_log_weights = this->cached_posterior_weights
                                                 .log_weights_from_children;

                    // Accumulate the previous weights
                    if (cum_log_weights.size() == 0) {
                        cum_log_weights = prev_log_weights;
                    }
                    else {
                        cum_log_weights =
                            cum_log_weights.array() + prev_log_weights.array();
                    }

                    this->cached_posterior_weights.time_step += 1;
                }

                // A node can be sampled multiple times before the lower time
                // step moves forward, we clear the cached log weights below
                // because we only accumulate the weights from the last
                // sampling step before the next lower time step.
                this->cached_posterior_weights.log_weights_from_children =
                    Eigen::MatrixXd(0, 0);
            }
        }

        Eigen::MatrixXd RandomVariableNode::get_cached_log_weights(
            int min_time_step_to_sample) const {
            Eigen::MatrixXd log_weights(0, 0);

            if (this->metadata->is_multitime()) {
                // Multitime nodes can have weights computed previously
                // cached when we perform forward inference.

                if (min_time_step_to_sample ==
                    this->cached_posterior_weights.time_step + 1) {

                    // Return the weights accumulated until the previous lower
                    // time step.
                    log_weights = this->cached_posterior_weights
                                      .cum_log_weights_from_children;
                }
            }

            return log_weights;
        }

        void RandomVariableNode::cache_current_log_weights(
            int min_time_step_to_sample,
            const RVNodePtr& child_node,
            const Eigen::MatrixXd& log_weights) {

            if (this->metadata->is_multitime()) {
                if (child_node->get_time_step() == min_time_step_to_sample) {
                    if (this->cached_posterior_weights.log_weights_from_children
                            .size() == 0) {
                        this->cached_posterior_weights
                            .log_weights_from_children = log_weights;
                    }
                    else {
                        // Sum of the log weights of all the children in the
                        // lower time step.
                        this->cached_posterior_weights
                            .log_weights_from_children =
                            this->cached_posterior_weights
                                .log_weights_from_children.array() +
                            log_weights.array();
                    }
                }
            }
        }

        NodePtrVec
        RandomVariableNode::get_children_in_range(int min_time_step,
                                                  int max_time_step) {
            NodePtrVec nodes;

            if (!this->children_per_time_step.empty()) {
                if (this->metadata->is_multitime()) {
                    for (int t = min_time_step; t <= max_time_step; t++) {
                        const auto& children =
                            this->children_per_time_step.at(t);
                        nodes.insert(
                            nodes.end(), children.begin(), children.end());
                    }
                }
                else {
                    // Children in the same time step as the node
                    if (min_time_step <= this->time_step &&
                        this->time_step <= max_time_step) {
                        const auto& children =
                            this->children_per_time_step.at(0);
                        nodes.insert(
                            nodes.end(), children.begin(), children.end());
                    }

                    // Children in the next time step
                    if (min_time_step <= this->time_step + 1 &&
                        this->time_step + 1 <= max_time_step) {
                        const auto& children =
                            this->children_per_time_step.at(1);
                        nodes.insert(
                            nodes.end(), children.begin(), children.end());
                    }
                }
            }

            return nodes;
        }

        Eigen::MatrixXd RandomVariableNode::get_segments_log_posterior_weights(
            int num_jobs, int max_time_step_to_sample) {
            Eigen::MatrixXd segments_weights(0, 0);

            if (!this->has_timer()) {
                // No weights if the node is not controlled by a timer
                return segments_weights;
            }

            const auto& central_state = shared_from_this();
            const auto& left_state = this->get_previous();
            const auto& right_state = this->time_step < max_time_step_to_sample
                                          ? this->get_next()
                                          : nullptr;

            const auto& central_timer = this->get_timer();
            const auto& left_last_timer =
                left_state ? left_state->get_timer() : nullptr;
            const auto& right_first_timer =
                right_state ? right_state->get_timer() : nullptr;

            // Last time step being sampled. This will be used to deal with
            // right segment truncation in the computation of the segment
            // posteriors.
            int last_time_step = max(this->time_step, max_time_step_to_sample);

            // Left segment
            if (left_state) {
                // Left segment
                Eigen::MatrixXd left_seg_weights =
                    left_last_timer->get_cpd()
                        ->get_left_segment_posterior_weights(left_last_timer,
                                                             right_state,
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

        bool RandomVariableNode::is_continuous() const {
            return this->cpd->is_continuous();
        }

        void RandomVariableNode::clear_cache() {
            this->cached_posterior_weights = PosteriorWeightsCache();
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

            this->children_per_time_step.clear();
            for (const auto& child : children) {
                const auto& rv_child =
                    dynamic_pointer_cast<RandomVariableNode>(child);

                if (this->metadata->is_multitime()) {
                    int t = rv_child->get_time_step();

                    if (this->children_per_time_step.size() <= t) {
                        this->children_per_time_step.resize(t + 1);
                    }

                    this->children_per_time_step.at(t).push_back(child);
                }
                else {
                    // It can only have children in the same time step and
                    // the next one. Therefore, we only need a vector of
                    // size 2 with the time steps relative to this node's
                    // time step.
                    if (this->children_per_time_step.empty()) {
                        this->children_per_time_step.resize(2);
                    }

                    if (rv_child->get_time_step() == this->time_step) {
                        this->children_per_time_step.at(0).push_back(child);
                    }
                    else {
                        // Next time step since back edges are not allowed
                        // in the implementation
                        this->children_per_time_step.at(1).push_back(child);
                    }
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

    } // namespace model
} // namespace tomcat
