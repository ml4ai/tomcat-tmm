#include "ParticleFilter.h"

#include <thread>

#include <boost/progress.hpp>

#include "distribution/Categorical.h"
#include "distribution/Distribution.h"
#include "pgm/TimerNode.h"
#include "utils/EigenExtensions.h"
#include "utils/Multithreading.h"

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        ParticleFilter::ParticleFilter() {}

        ParticleFilter::ParticleFilter(
            const DynamicBayesNet& dbn,
            int num_particles,
            const std::shared_ptr<gsl_rng>& random_generator,
            int num_jobs)
            : original_dbn(dbn), num_particles(num_particles) {

            this->create_template_dbn();
            this->random_generators_per_job =
                split_random_generator(random_generator, num_jobs);
        }

        ParticleFilter::~ParticleFilter() {}

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------

        void ParticleFilter::create_template_dbn() {
            this->template_dbn = this->original_dbn.clone(false);
            this->template_dbn.unroll(LAST_TEMPLATE_TIME_STEP + 1, true);
            this->template_dbn.mirror_parameter_nodes_from(this->original_dbn);

            // We freeze the distributions to optimize computations. This will
            // transform the list of distributions matrices of enumerated
            // probabilities in bounded discrete distributions.
            for (const auto& node : this->template_dbn.get_nodes()) {
                if (!node->get_metadata()->is_parameter()) {
                    RVNodePtr rv_node =
                        dynamic_pointer_cast<RandomVariableNode>(node);
                    rv_node->get_cpd()->freeze_distributions(0);
                    data_node_labels.insert(
                        rv_node->get_metadata()->get_label());
                }
            }

            // Single time nodes will be marginalized from the bottom to the
            // top.
            for (const auto& node :
                 this->template_dbn.get_nodes_topological_order(false)) {
                if (!node->get_metadata()->is_replicable() &&
                    !node->get_metadata()->is_parameter()) {

                    RVNodePtr rv_node =
                        dynamic_pointer_cast<RandomVariableNode>(node);
                    const auto& metadata = node->get_metadata();

                    this->marginal_nodes.push_back(
                        dynamic_pointer_cast<RandomVariableNode>(node));
                    this->marginal_set.insert(metadata->get_label());
                    this->marginal_posterior_weights[metadata->get_label()] =
                        Eigen::MatrixXd::Zero(this->num_particles,
                                              metadata->get_cardinality());
                }
            }
        }

        pair<EvidenceSet, EvidenceSet>
        ParticleFilter::generate_particles(const EvidenceSet& new_data) {
            unique_ptr<boost::progress_display> progress;

            if (this->show_progress) {
                progress = make_unique<boost::progress_display>(
                    new_data.get_time_steps());
            }

            EvidenceSet particles;
            EvidenceSet marginals;
            int initial_time_step = this->last_time_step + 1;
            int final_time_step =
                this->last_time_step + new_data.get_time_steps();
            for (int t = initial_time_step; t <= final_time_step; t++) {
                this->elapse(new_data, t);
                particles.hstack(this->resample(new_data, t));
                marginals.hstack(this->apply_rao_blackwellization(t));
                this->move_particles_back_in_time(t);

                if (this->show_progress) {
                    ++(*progress);
                }
            }

            this->last_time_step = final_time_step;

            return make_pair(particles, marginals);
        }

        void ParticleFilter::elapse(const EvidenceSet& new_data,
                                    int time_step) {
            int template_time_step = min(time_step, LAST_TEMPLATE_TIME_STEP);

            for (const auto& node :
                 this->template_dbn.get_data_nodes_in_topological_order_at(
                     template_time_step)) {
                const string& node_label = node->get_metadata()->get_label();

                if (new_data.has_data_for(node_label)) {
                    const auto& data = new_data[node_label];
                    Eigen::MatrixXd observation(1, data.get_shape()[0]);
                    observation.row(0) =
                        data.depth(0, time_step - this->last_time_step - 1);
                    observation = observation.replicate(this->num_particles, 1);
                    node->set_assignment(observation);

                    if (!node->get_metadata()->is_replicable()) {
                        // Freeze node to skip resampling its assignments as
                        // they are all the same and do not change over time.
                        node->freeze();
                    }
                }
                else {
                    //                    if (node_label == "State") {
                    //                        Eigen::MatrixXd states(5, 5);
                    //                        states << 3, 3, 1, 3, 2, 2, 3, 1,
                    //                        3, 3, 1, 3, 3, 2, 1,
                    //                            3, 3, 2, 3, 1, 3, 1, 2, 2, 1;
                    //                        states.transposeInPlace();
                    //                        states.array() -= 1;
                    //                        node->set_assignment(states.col(time_step));
                    //                        continue;
                    //                    }
                    //                    else if (node_label == "Fixed") {
                    //                        Eigen::MatrixXd fixed(1, 5);
                    //                        fixed << 1, 3, 3, 3, 2;
                    //                        fixed.transposeInPlace();
                    //                        fixed.array() -= 1;
                    //                        node->set_assignment(fixed);
                    //                        continue;
                    //                    }

                    Eigen::MatrixXd samples;

                    if (node->has_timer() && time_step > 0) {
                        // Sample from the joint distribution of timed
                        // controlled node and duration
                        const auto& left_timer =
                            dynamic_pointer_cast<TimerNode>(
                                node->get_timer()->get_previous());
                        Eigen::MatrixXd duration_weights =
                            left_timer->get_cpd()
                                ->get_left_segment_posterior_weights(
                                    left_timer,
                                    this->last_left_segment_distribution_indices
                                        [node_label],
                                    nullptr,
                                    time_step,
                                    time_step,
                                    this->random_generators_per_job.size());

                        samples = node->get_cpd()->sample_from_posterior(
                            this->random_generators_per_job,
                            duration_weights,
                            node);
                    }
                    else if (node->get_metadata()->is_timer()) {
                        // This will increment the timer by one time step if the
                        // new sampled state is the same as the one sampled in
                        // the previous time step. Otherwise, the timer will be
                        // set to zero.
                        samples = node->sample_from_posterior(
                            this->random_generators_per_job);

                        // Store the distributions of the most recent left
                        // segment
                        auto timer = dynamic_pointer_cast<TimerNode>(node);
                        Eigen::VectorXi distribution_indices =
                            timer->get_cpd()->get_indexed_distribution_indices(
                                timer->get_parents(), this->num_particles);
                        const string& timed_node_label =
                            timer->get_controlled_node()
                                ->get_metadata()
                                ->get_label();

                        if (time_step == 0) {
                            this->last_left_segment_distribution_indices
                                [timed_node_label] = distribution_indices;
                        }
                        else {
                            Eigen::VectorXi& curr_indices =
                                this->last_left_segment_distribution_indices
                                    [timed_node_label];
                            Eigen::VectorXi new_segments =
                                (samples.array() == 0).cast<int>();
                            Eigen::VectorXi open_segments =
                                1 - new_segments.array();
                            // Update the indices of particles in which a new
                            // segment started.
                            curr_indices = move(curr_indices.array() *
                                                    open_segments.array() +
                                                distribution_indices.array() *
                                                    new_segments.array());
                        }
                    }
                    else {
                        samples = node->sample(this->random_generators_per_job,
                                               this->num_particles);
                    }

                    if (node->get_metadata()->is_timer()) {
                        dynamic_pointer_cast<TimerNode>(node)
                            ->set_forward_assignment(samples);
                    }
                    else {
                        node->set_assignment(samples);
                    }
                }
            }
        }

        void ParticleFilter::update_timer_forward_assignment(
            const TimerNodePtr& timer) {
            Eigen::MatrixXd forward_assignment =
                Eigen::MatrixXd::Zero(this->num_particles, 1);
            if (auto prev_timer =
                    dynamic_pointer_cast<TimerNode>(timer->get_previous(1))) {
                Eigen::MatrixXd previous_backward_assignment =
                    prev_timer->get_backward_assignment();
                Eigen::MatrixXd previous_forward_assignment =
                    prev_timer->get_forward_assignment();

                Eigen::MatrixXd open_segments =
                    (previous_backward_assignment.array() != 0).cast<double>();
                forward_assignment = (previous_forward_assignment.array() + 1) *
                                     open_segments.array();
            }
            timer->set_forward_assignment(forward_assignment);
        }

        EvidenceSet ParticleFilter::resample(const EvidenceSet& new_data,
                                             int time_step) {
            int template_time_step = min(time_step, LAST_TEMPLATE_TIME_STEP);
            Eigen::MatrixXd sampled_particles;

            if (!new_data.empty()) {
                // If no data is provided, all particles are kept. No particle
                // resampling is necessary.
                Eigen::VectorXd log_weights =
                    Eigen::VectorXd::Zero(this->num_particles);
                bool has_data_at_time_step = false;

                for (const auto& node_label : new_data.get_node_labels()) {
                    if (this->template_dbn.has_node_with_label(node_label)) {
                        const auto& node = this->template_dbn.get_node(
                            node_label, template_time_step);
                        if (node) {
                            log_weights.array() +=
                                node->get_pdfs(
                                        this->random_generators_per_job.size(),
                                        0)
                                    .array()
                                    .log();
                            has_data_at_time_step = true;
                        }
                    }
                }

                if (has_data_at_time_step) {
                    Eigen::VectorXd probabilities =
                        log_weights.array() - log_weights.maxCoeff();
                    probabilities = probabilities.array().exp();
                    probabilities = probabilities.array() / probabilities.sum();

                    sampled_particles =
                        Categorical(move(probabilities))
                            .sample_many(this->random_generators_per_job,
                                         this->num_particles,
                                         0);

                    //                    Eigen::MatrixXd particles(4, 5);
                    //                    particles << 1, 1, 4, 3, 5, 4, 1, 2,
                    //                    1, 1, 3, 3, 4, 5, 5, 5,
                    //                        2, 3, 4, 5;
                    //                    particles.transposeInPlace();
                    //                    particles.array() -= 1;
                    //                    sampled_particles =
                    //                    particles.col(time_step - 1);
                }
            }

            EvidenceSet particles;
            for (const auto& node_label : this->data_node_labels) {
                if (!EXISTS(node_label, this->marginal_set)) {
                    particles.add_data(
                        node_label,
                        Tensor3::constant(1, this->num_particles, 1, NO_OBS));
                }
            }

            RVNodePtrVec nodes =
                this->template_dbn.get_data_nodes(template_time_step);
            for (const auto& node :
                 this->template_dbn.get_single_time_nodes()) {
                if (node->get_metadata()->get_initial_time_step() < time_step) {
                    nodes.push_back(node);
                }
            }

            for (const auto& node : nodes) {
                const string& node_label = node->get_metadata()->get_label();

                if (EXISTS(node_label, this->marginal_set)) {
                    // These nodes will be sampled from their updated
                    // posterior later. Here we just rearrange marginal
                    // posterior weights
                    if (sampled_particles.size() > 0 && !node->is_frozen()) {
                        auto& posterior_weights =
                            this->marginal_posterior_weights[node_label];
                        Eigen::MatrixXd filtered_weights(
                            posterior_weights.rows(), posterior_weights.cols());

                        for (int i = 0; i < this->num_particles; i++) {
                            int particle = sampled_particles(i, 0);
                            filtered_weights.row(i) =
                                posterior_weights.row(particle);
                        }

                        posterior_weights = move(filtered_weights);

                        if (EXISTS(
                                node_label,
                                this->last_left_segment_marginal_nodes_distribution_indices)) {
                            for (
                                auto& [timer_label, distribution_indices] :
                                this->last_left_segment_marginal_nodes_distribution_indices
                                    [node_label]) {
                                Eigen::VectorXi filtered_indices(
                                    distribution_indices.size());

                                for (int i = 0; i < this->num_particles; i++) {
                                    int particle = sampled_particles(i, 0);
                                    filtered_indices(i) =
                                        distribution_indices(particle);
                                }

                                distribution_indices = move(filtered_indices);
                            }
                        }
                    }
                    continue;
                }

                Eigen::MatrixXd samples;
                if (node->get_metadata()->is_timer()) {
                    samples = dynamic_pointer_cast<TimerNode>(node)
                                  ->get_forward_assignment();
                }
                else {
                    samples = node->get_assignment();
                }

                Eigen::MatrixXd filtered_samples(this->num_particles,
                                                 samples.cols());
                if (node->is_frozen() || sampled_particles.size() == 0) {
                    filtered_samples = move(samples);
                }
                else {
                    // We need to update the samples from nodes in the previous
                    // time step too so that distributions are correctly
                    // addressed in the marginalization process.
                    Eigen::MatrixXd filtered_previous_samples;
                    if (node->get_previous()) {
                        filtered_previous_samples = Eigen::MatrixXd(
                            this->num_particles, samples.cols());
                    }

                    // Rearrange the posterior weights computed for marginal
                    // nodes per particle.
                    for (int i = 0; i < this->num_particles; i++) {
                        int particle = sampled_particles(i, 0);
                        filtered_samples.row(i) = samples.row(particle);

                        if (node->get_previous()) {
                            if (node->get_metadata()->is_timer()) {
                                filtered_previous_samples.row(i) =
                                    dynamic_pointer_cast<TimerNode>(
                                        node->get_previous())
                                        ->get_forward_assignment()
                                        .row(particle);
                            }
                            else {
                                filtered_previous_samples.row(i) =
                                    node->get_previous()->get_assignment().row(
                                        particle);
                            }
                        }
                    }

                    if (node->get_previous()) {
                        if (node->get_metadata()->is_timer()) {
                            dynamic_pointer_cast<TimerNode>(
                                node->get_previous())
                                ->set_forward_assignment(
                                    move(filtered_previous_samples));
                        }
                        else {
                            node->get_previous()->set_assignment(
                                move(filtered_previous_samples));
                        }
                    }

                    if (node->has_timer()) {
                        // Rearrange left segment distribution indices
                        auto& left_segment_distribution_indices =
                            this->last_left_segment_distribution_indices
                                [node_label];
                        Eigen::VectorXi filtered_indices(
                            left_segment_distribution_indices.size());

                        for (int i = 0; i < this->num_particles; i++) {
                            int particle = sampled_particles(i, 0);
                            filtered_indices(i) =
                                left_segment_distribution_indices(particle);
                        }

                        left_segment_distribution_indices =
                            move(filtered_indices);
                    }
                }

                // Save particles
                Tensor3 filtered_samples_tensor(filtered_samples);
                if (filtered_samples.cols() > 1) {
                    // If the sample has more than one dimension, we
                    // move the dimension to the depth axis of the
                    // tensor because the column is reserved for the
                    // time dimension.
                    filtered_samples_tensor.reshape(
                        filtered_samples.cols(), this->num_particles, 1);
                }

                particles.add_data(node_label, filtered_samples_tensor);

                if (node->get_metadata()->is_timer()) {
                    dynamic_pointer_cast<TimerNode>(node)
                        ->set_forward_assignment(move(filtered_samples));
                }
                else {
                    node->set_assignment(move(filtered_samples));
                }
            }

            return particles;
        }

        EvidenceSet ParticleFilter::apply_rao_blackwellization(int time_step) {
            EvidenceSet marginals;
            for (const auto& node : this->marginal_nodes) {

                const auto& metadata = node->get_metadata();
                const string& node_label = metadata->get_label();

                if (node->is_frozen()) {
                    // Observations were provided for this node so there's no
                    // need to sample from it.
                    Eigen::VectorXd probabilities =
                        Eigen::VectorXd::Zero(metadata->get_cardinality());
                    probabilities(node->get_assignment()(0, 0)) = 1;
                    marginals.add_data(node_label, Tensor3(probabilities));
                    continue;
                }

                if (metadata->get_initial_time_step() <= time_step) {

                    // Accumulates weights for each particle and child node. If
                    // the child node is a timer, weights are accumulated
                    // separately for open and closed segments.
                    Eigen::MatrixXd child_log_weights = Eigen::MatrixXd::Zero(
                        this->num_particles, metadata->get_cardinality());
                    Eigen::MatrixXd open_child_segment_log_weights =
                        Eigen::MatrixXd::Zero(this->num_particles,
                                              metadata->get_cardinality());
                    Eigen::MatrixXd closed_child_segment_log_weights =
                        Eigen::MatrixXd::Zero(this->num_particles,
                                              metadata->get_cardinality());

                    int children_template_time_step =
                        min(time_step, LAST_TEMPLATE_TIME_STEP);

                    auto& cum_weights =
                        this->marginal_posterior_weights[node_label];

                    for (const auto& child :
                         node->get_children(children_template_time_step)) {

                        if (child->get_metadata()->is_timer()) {
                            const auto& timer =
                                dynamic_pointer_cast<TimerNode>(child);

                            // Get the list of distributions (one per particle)
                            // when this parent of the timer node is 0. This is
                            // the baseline for us to get the other
                            // distributions (one for each possible value the
                            // node can take).
                            Eigen::MatrixXd current_assignment =
                                node->get_assignment();
                            node->set_assignment(Eigen::MatrixXd::Zero(
                                this->num_particles,
                                current_assignment.cols()));
                            Eigen::VectorXi new_distribution_indices =
                                child->get_cpd()
                                    ->get_indexed_distribution_indices(
                                        child->get_parents(),
                                        this->num_particles);
                            node->set_assignment(current_assignment);

                            // Identify particles in which a new segment begins.
                            // We need to update the distributions of the last
                            // segment for those particles.
                            Eigen::VectorXi new_segments =
                                (timer->get_forward_assignment().array() == 0)
                                    .cast<int>();
                            Eigen::VectorXi open_segments =
                                1 - new_segments.array();

                            if (EXISTS(
                                    node_label,
                                    last_left_segment_marginal_nodes_distribution_indices) &&
                                EXISTS(
                                    timer->get_metadata()->get_label(),
                                    last_left_segment_marginal_nodes_distribution_indices
                                        [node_label])) {
                                auto& curr_distribution_indices =
                                    last_left_segment_marginal_nodes_distribution_indices
                                        [node_label]
                                        [timer->get_metadata()->get_label()];

                                curr_distribution_indices =
                                    move(curr_distribution_indices.array() *
                                             open_segments.array() +
                                         new_distribution_indices.array() *
                                             new_segments.array());
                            }
                            else {
                                last_left_segment_marginal_nodes_distribution_indices
                                    [node_label]
                                    [timer->get_metadata()->get_label()] =
                                        new_distribution_indices;
                            }

                            if (time_step > 0) {
                                // Compute posterior weight for the left segment
                                // timer. There's no need to compute the
                                // posterior weights for the central timer
                                // because p(duration
                                // >= 0 | Pa(timer)) is always 1 regardless of
                                // the value of the timer's parent node.
                                const auto& left_segment_timer =
                                    dynamic_pointer_cast<TimerNode>(
                                        timer->get_previous());
                                const auto& distribution_indices =
                                    last_left_segment_marginal_nodes_distribution_indices
                                        [node_label]
                                        [timer->get_metadata()->get_label()];

                                Eigen::MatrixXd child_segment_weights =
                                    timer->get_cpd()
                                        ->get_particle_timer_posterior_weights(
                                            distribution_indices,
                                            node,
                                            left_segment_timer,
                                            this->random_generators_per_job
                                                .size());

                                // Weights on particles with open/closed segment
                                // amd 1 in the others.
                                open_child_segment_log_weights.array() +=
                                    (((child_segment_weights.array().colwise() *
                                       open_segments.array().cast<double>())
                                          .colwise() +
                                      new_segments.array().cast<double>())
                                         .array() +
                                     EPSILON)
                                        .log();
                                closed_child_segment_log_weights.array() +=
                                    (((child_segment_weights.array().colwise() *
                                       new_segments.array().cast<double>())
                                          .colwise() +
                                      open_segments.array().cast<double>())
                                         .array() +
                                     EPSILON)
                                        .log();
                            }
                        }
                        else {
                            Eigen::MatrixXd child_weights =
                                child->get_cpd()->get_posterior_weights(
                                    child->get_parents(),
                                    node,
                                    child,
                                    this->random_generators_per_job.size());

                            child_log_weights.array() +=
                                (child_weights.array() + EPSILON).log();
                        }
                    }

                    // Accumulate weights
                    this->marginal_posterior_weights[node_label].array() +=
                        child_log_weights.array() +
                        closed_child_segment_log_weights.array();

                    // Compute weights for the current posteriors per particle.
                    // Open segments are considered here but they are only
                    // accumulates when they become closed, which is when the
                    // semi-Markov model jumps to a new sequence of states in a
                    // different segment.
                    Eigen::MatrixXd weights_per_particle =
                        this->marginal_posterior_weights[node_label].array() +
                        open_child_segment_log_weights.array();
                    weights_per_particle.colwise() -=
                        weights_per_particle.rowwise().maxCoeff();
                    weights_per_particle = weights_per_particle.array().exp();
                    Eigen::VectorXd sum_per_row =
                        weights_per_particle.rowwise().sum();
                    weights_per_particle =
                        weights_per_particle.array().colwise() /
                        sum_per_row.array();

                    // Compute posterior, calculate estimate based on the
                    // posterior of all the particles and sample a new value
                    // from that posterior.
                    auto posteriors = node->get_cpd()->get_posterior(
                        weights_per_particle,
                        node,
                        this->random_generators_per_job.size());
                    Eigen::VectorXd probabilities =
                        Eigen::VectorXd::Zero(metadata->get_cardinality());

                    int i = 0;
                    for (const auto& posterior : posteriors) {
                        probabilities.array() +=
                            posterior->get_values(0).array();
                        node->set_assignment(
                            i++,
                            0,
                            posterior->sample(
                                this->random_generators_per_job[0], 0)(0));
                    }
                    probabilities.array() /= probabilities.sum();
                    marginals.add_data(node_label, Tensor3(probabilities));

                    //                    Eigen::MatrixXd fixed(4, 5);
                    //                    fixed << 3, 3, 3, 3, 3, 3, 3, 1, 3, 3,
                    //                    2, 3, 2, 3, 3, 1, 1,
                    //                        1, 1, 1;
                    //                    fixed.transposeInPlace();
                    //                    fixed.array() -= 1;
                    //                    node->set_assignment(fixed.col(time_step
                    //                    - 1));
                }
                else {
                    // Prior
                    marginals.add_data(
                        metadata->get_label(),
                        Tensor3(
                            node->get_cpd()->get_distributions()[0]->get_values(
                                0)));
                }
            }

            return marginals;
        }

        void ParticleFilter::move_particles_back_in_time(int time_step) {
            if (time_step >= LAST_TEMPLATE_TIME_STEP) {
                for (const auto& node : this->template_dbn.get_data_nodes(
                         LAST_TEMPLATE_TIME_STEP)) {
                    auto previous_node = this->template_dbn.get_node(
                        node->get_metadata()->get_label(),
                        LAST_TEMPLATE_TIME_STEP - 1);
                    // We save the assignment in the node at time step 1
                    // because next time we elapse, nodes in time step 2
                    // will be sampled. Therefore, the samples generated
                    // here will be assigned to the proper parent of the
                    // nodes at time step 2.
                    if (previous_node) {
                        previous_node->set_assignment(node->get_assignment());

                        if (node->get_metadata()->is_timer()) {
                            dynamic_pointer_cast<TimerNode>(previous_node)
                                ->set_forward_assignment(
                                    dynamic_pointer_cast<TimerNode>(node)
                                        ->get_forward_assignment());
                        }
                    }
                }
            }
        }

        EvidenceSet ParticleFilter::forward_particles(int num_time_steps) {
            EvidenceSet particles;

            if (num_time_steps > 0) {
                int template_time_step =
                    min(this->last_time_step, LAST_TEMPLATE_TIME_STEP);

                RVNodePtrVec nodes =
                    this->template_dbn.get_data_nodes(template_time_step);
                for (const auto& node :
                     this->template_dbn.get_single_time_nodes()) {
                    if (node->get_metadata()->get_initial_time_step() <
                        template_time_step) {
                        nodes.push_back(node);
                    }
                }

                // Save last particles generated to be restored after we
                // generate samples forward
                EvidenceSet last_particles;
                for (const auto& node : nodes) {
                    last_particles.add_data(node->get_metadata()->get_label(),
                                            node->get_assignment());
                }

                EvidenceSet empty_set;
                int initial_time_step = this->last_time_step + 1;
                int final_time_step = this->last_time_step + num_time_steps;
                for (int t = initial_time_step; t <= final_time_step; t++) {
                    this->elapse(empty_set, t);
                    particles.hstack(this->resample(empty_set, t));
                }

                // Restore particles
                for (const auto& node : nodes) {
                    node->set_assignment(
                        last_particles[node->get_metadata()->get_label()](0,
                                                                          0));
                }
            }

            return particles;
        }

        void ParticleFilter::clear_cache() { this->last_time_step = -1; }

        Eigen::MatrixXd
        ParticleFilter::shuffle_rows(const Eigen::MatrixXd& matrix,
                                     const Eigen::VectorXi& rows) const {

            Eigen::MatrixXd shuffled_matrix(matrix.rows(), matrix.cols());
            mutex shuffled_matrix_mutex;

            int num_jobs = this->random_generators_per_job.size();
            if (num_jobs == 1) {
                this->run_shuffle_rows_thread(make_pair(0, matrix.rows()),
                                              matrix,
                                              shuffled_matrix,
                                              rows,
                                              shuffled_matrix_mutex);
            }
            else {
                auto processing_blocks =
                    get_parallel_processing_blocks(num_jobs, matrix.rows());
                vector<thread> threads;
                for (const auto& processing_block : processing_blocks) {
                    thread shuffle_thread(
                        &ParticleFilter::run_shuffle_rows_thread,
                        this,
                        processing_block,
                        matrix,
                        ref(shuffled_matrix),
                        rows,
                        ref(shuffled_matrix_mutex));

                    threads.push_back(move(shuffle_thread));
                }

                for (auto& shuffled_thread : threads) {
                    shuffled_thread.join();
                }
            }

            return shuffled_matrix;
        }

        void ParticleFilter::run_shuffle_rows_thread(
            const pair<int, int>& processing_block,
            const Eigen::MatrixXd& original_matrix,
            Eigen::MatrixXd& shuffled_matrix,
            const Eigen::VectorXi& rows,
            mutex& shuffled_matrix_mutex) const {

            int initial_row = processing_block.first;
            int num_rows = processing_block.second;

            Eigen::MatrixXd partially_shuffled_matrix(num_rows,
                                                      original_matrix.cols());
            for (int i = initial_row; i < initial_row + num_rows; i++) {
                int idx = rows(i);
                partially_shuffled_matrix.row(i) = original_matrix.row(idx);
            }

            scoped_lock lock(shuffled_matrix_mutex);
            shuffled_matrix.block(
                initial_row, 0, num_rows, original_matrix.cols()) =
                partially_shuffled_matrix;
        }

        //----------------------------------------------------------------------
        // Getters & Setters
        //----------------------------------------------------------------------

        void ParticleFilter::set_show_progress(bool show_progress) {
            this->show_progress = show_progress;
        }

    } // namespace model
} // namespace tomcat
