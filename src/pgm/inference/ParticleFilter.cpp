#include "ParticleFilter.h"

#include <thread>

#include <boost/progress.hpp>

#include "distribution/Categorical.h"
#include "distribution/Distribution.h"
#include "pgm/TimerNode.h"
#include "pipeline/estimation/SamplerEstimator.h"
#include "utils/EigenExtensions.h"
#include "utils/Multithreading.h"

namespace tomcat {
    namespace model {

        using namespace std;
        using namespace multithread;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        //        ParticleFilter::ParticleFilter() {}

        ParticleFilter::ParticleFilter(
            const DynamicBayesNet& dbn,
            int num_particles,
            const std::shared_ptr<gsl_rng>& random_generator,
            int num_jobs)
            : original_dbn(dbn), num_particles(num_particles),
              thread_pool(num_jobs) {

            this->create_template_dbn();
            this->random_generators_per_job =
                split_random_generator(random_generator, num_jobs);
            this->processing_blocks =
                get_parallel_processing_blocks(num_jobs, num_particles);
        }

        ParticleFilter::~ParticleFilter() {}

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------

        void ParticleFilter::create_template_dbn() {
            this->template_dbn = this->original_dbn.clone(false);
            this->template_dbn.unroll(LAST_TEMPLATE_TIME_STEP + 1, true);
            this->template_dbn.mirror_parameter_nodes_from(this->original_dbn);

            for (const auto& node : this->template_dbn.get_nodes()) {
                if (!node->get_metadata()->is_parameter()) {
                    RVNodePtr rv_node =
                        dynamic_pointer_cast<RandomVariableNode>(node);

                    // This will speed sampling
                    rv_node->get_cpd()->freeze_distributions(0);

                    data_node_labels.insert(
                        rv_node->get_metadata()->get_label());
                    if (rv_node->has_timer()) {
                        this->time_controlled_node_set.insert(
                            rv_node->get_metadata()->get_label());
                    }

                    // Initialize nodes' assignment
                    int sample_size =
                        rv_node->get_metadata()->get_sample_size();
                    rv_node->set_assignment(Eigen::MatrixXd::Zero(
                        this->num_particles, sample_size));
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
                    this->cum_marginal_posterior_log_weights
                        [metadata->get_label()] = Eigen::MatrixXd::Zero(
                        this->num_particles, metadata->get_cardinality());
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
            if (new_data.is_event_based()) {
                final_time_step =
                    min(final_time_step, new_data.get_num_events_for(0) - 1);
            }

            vector<future<void>> futures(this->processing_blocks.size());
            for (int t = initial_time_step; t <= final_time_step; t++) {
                for (int i = 0; i < this->processing_blocks.size(); i++) {
                    futures[i] = this->thread_pool.submit(
                        bind(&ParticleFilter::predict,
                             this,
                             ref(new_data),
                             t,
                             ref(this->processing_blocks[i]),
                             ref(this->random_generators_per_job[i])));
                }

                // Wait for all the threads to elapse and weigh particles
                for (auto& future : futures) {
                    future.get();
                }

                // Prepare the discrete distribution of particles.
                shared_ptr<gsl_ran_discrete_t> ptable(gsl_ran_discrete_preproc(
                    weights.rows(), weights.col(0).data()));

                for (int i = 0; i < this->processing_blocks.size(); i++) {
                    // This will resample and compute RBW
                    futures[i] = this->thread_pool.submit(
                        bind(&ParticleFilter::resample,
                             this,
                             ref(new_data),
                             t,
                             ref(this->processing_blocks[i])));
                }

                //                this->elapse(new_data, t);
                //                Eigen::VectorXi sampled_particles =
                //                    this->weigh_and_sample_particles(t,
                //                    new_data);
                //                EvidenceSet resampled_particles =
                //                    this->resample(new_data, t,
                //                    sampled_particles);
                //                marginals.hstack(this->apply_rao_blackwellization(
                //                    t, resampled_particles,
                //                    sampled_particles));
                //                particles.hstack(resampled_particles);
                //                this->update_left_segment_distribution_indices(t);
                //                this->move_particles_back_in_time(t);

                if (this->show_progress) {
                    ++(*progress);
                }
            }

            // Wait for all blocks to be processed
            for (auto& future : futures) {
                future.get();
            }

            this->last_time_step = final_time_step;

            return make_pair(particles, marginals);
        }

        void
        ParticleFilter::predict(const EvidenceSet& new_data,
                                int time_step,
                                const ProcessingBlock& processing_block,
                                std::shared_ptr<gsl_rng>& random_generator) {

            int template_time_step = min(time_step, LAST_TEMPLATE_TIME_STEP);

            for (auto& node :
                 this->template_dbn.get_data_nodes_in_topological_order_at(
                     template_time_step)) {
                const string& node_label = node->get_metadata()->get_label();

                if (new_data.has_data_for(node_label)) {
                    Eigen::MatrixXd data = new_data[node_label].depth(
                        0, time_step - this->last_time_step - 1);
                    this->fix_evidence(node, data, processing_block);
                }
                else {
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
                                    processing_block);

                        samples = node->get_cpd()->sample_from_posterior(
                            random_generator,
                            duration_weights,
                            node,
                            processing_block);
                    }
                    else if (node->get_metadata()->is_timer()) {
                        // This will increment the timer by one time step if the
                        // new sampled state is the same as the one sampled in
                        // the previous time step. Otherwise, the timer will be
                        // set to zero.
                        samples = node->sample_from_posterior(random_generator,
                                                              processing_block);
                    }
                    else {
                        samples =
                            node->sample(random_generator, processing_block);
                    }

                    if (node->get_metadata()->is_timer()) {
                        dynamic_pointer_cast<TimerNode>(node)
                            ->set_forward_assignment(samples, processing_block);
                    }
                    else {
                        node->set_assignment(samples, processing_block);
                    }
                }
            }
        }

        void
        ParticleFilter::fix_evidence(const RVNodePtr& node,
                                     const Eigen::MatrixXd& data,
                                     const ProcessingBlock& processing_block) {

            int initial_row = processing_block.first;
            int num_rows = processing_block.second;

            Eigen::MatrixXd observation = data;
            node.set_assignment(observation.replicate(num_rows, 1),
                                processing_block);
        }

        EvidenceSet
        ParticleFilter::resample(const EvidenceSet& new_data,
                                 int time_step,
                                 const Eigen::VectorXi& sampled_particles) {
            int template_time_step = min(time_step, LAST_TEMPLATE_TIME_STEP);

            // Prepare set of estimates with the label of the nodes in the DBN
            EvidenceSet particles;
            for (const auto& node_label : this->data_node_labels) {
                if (!EXISTS(node_label, this->marginal_set)) {
                    particles.add_data(
                        node_label,
                        Tensor3::constant(1, this->num_particles, 1, NO_OBS));
                }
            }

            // Node in the current time step + single time nodes will be
            // processed
            RVNodePtrVec nodes =
                this->template_dbn.get_data_nodes(template_time_step);
            for (const auto& node :
                 this->template_dbn.get_single_time_nodes()) {
                if (node->get_metadata()->get_initial_time_step() < time_step &&
                    EXISTS(node->get_metadata()->get_label(),
                           this->marginal_set)) {
                    nodes.push_back(node);
                }
            }

            for (const auto& node : nodes) {
                const string& node_label = node->get_metadata()->get_label();

                this->shuffle_node_and_previous(node, sampled_particles);

                if (EXISTS(node_label, this->marginal_set)) {
                    // The estimates for these nodes are computed in an exact
                    // manner at a later stage.
                    this->shuffle_marginal_posterior_weights(node,
                                                             sampled_particles);
                    continue;
                }

                if (node->has_timer() && time_step > 0) {
                    this->shuffle_timed_node_left_segment_distributions(
                        node, sampled_particles);
                }

                Eigen::MatrixXd filtered_samples;
                if (node->get_metadata()->is_timer()) {
                    filtered_samples = dynamic_pointer_cast<TimerNode>(node)
                                           ->get_forward_assignment();
                }
                else {
                    filtered_samples = node->get_assignment();
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
            }

            return particles;
        }

        Eigen::VectorXi ParticleFilter::weigh_and_sample_particles(
            int time_step, const EvidenceSet& new_data) const {
            Eigen::VectorXi sampled_particles;

            if (!new_data.empty()) {
                // If no data is provided, all particles are kept. No particle
                // resampling is necessary.

                int template_time_step =
                    min(time_step, LAST_TEMPLATE_TIME_STEP);
                Eigen::VectorXd log_weights =
                    Eigen::VectorXd::Zero(this->num_particles);
                bool has_data_at_time_step = false;

                for (const auto& node_label : new_data.get_node_labels()) {
                    if (this->template_dbn.has_node_with_label(node_label)) {
                        const auto& node = this->template_dbn.get_node(
                            node_label, template_time_step);
                        const auto& data = new_data[node_label];
                        double value = data.depth(
                            0, time_step - this->last_time_step - 1)(0, 0);
                        if (value == NO_OBS) {
                            // No observation for the node at this time step
                            continue;
                        }
                        if (node) {
                            log_weights.array() +=
                                (node->get_pdfs(
                                         this->random_generators_per_job.size(),
                                         0)
                                     .array() +
                                 EPSILON)
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
                                         0)
                            .col(0)
                            .cast<int>();
                }
            }

            return sampled_particles;
        }

        void ParticleFilter::shuffle_marginal_posterior_weights(
            const RVNodePtr& node, const Eigen::VectorXi& sampled_particles) {

            const string& node_label = node->get_metadata()->get_label();
            if (sampled_particles.size() > 0 && !node->is_frozen()) {
                auto& posterior_weights =
                    this->cum_marginal_posterior_log_weights[node_label];

                posterior_weights = move(
                    this->shuffle_rows(posterior_weights, sampled_particles));

                if (EXISTS(
                        node_label,
                        this->last_left_segment_marginal_nodes_distribution_indices)) {
                    for (
                        auto& [timer_label, distribution_indices] :
                        this->last_left_segment_marginal_nodes_distribution_indices
                            [node_label]) {

                        distribution_indices = move(this->shuffle_rows(
                            distribution_indices, sampled_particles));
                    }
                }
            }
        }

        void ParticleFilter::shuffle_node_and_previous(
            const RVNodePtr& node, const Eigen::VectorXi& sampled_particles) {
            if (!node->is_frozen() && sampled_particles.size() > 0) {

                if (node->get_metadata()->is_timer()) {
                    const auto& timer = dynamic_pointer_cast<TimerNode>(node);

                    timer->set_forward_assignment(this->shuffle_rows(
                        timer->get_forward_assignment(), sampled_particles));

                    if (timer->get_previous()) {
                        const auto& previous_timer =
                            dynamic_pointer_cast<TimerNode>(
                                timer->get_previous());
                        previous_timer->set_forward_assignment(
                            this->shuffle_rows(
                                previous_timer->get_forward_assignment(),
                                sampled_particles));
                    }
                }
                else {
                    node->set_assignment(this->shuffle_rows(
                        node->get_assignment(), sampled_particles));

                    if (node->get_previous()) {
                        const auto& previous_node = node->get_previous();
                        previous_node->set_assignment(
                            this->shuffle_rows(previous_node->get_assignment(),
                                               sampled_particles));
                    }
                }
            }
        }

        void ParticleFilter::shuffle_timed_node_left_segment_distributions(
            const RVNodePtr& node, const Eigen::VectorXi& sampled_particles) {
            if (sampled_particles.size() > 0) {
                // Rearrange left segment distribution indices
                auto& left_segment_distribution_indices =
                    this->last_left_segment_distribution_indices
                        [node->get_metadata()->get_label()];

                left_segment_distribution_indices = move(this->shuffle_rows(
                    left_segment_distribution_indices, sampled_particles));
            }
        }

        EvidenceSet ParticleFilter::apply_rao_blackwellization(
            int time_step,
            EvidenceSet& particles,
            const Eigen::VectorXi& sampled_particles) {
            EvidenceSet marginals;

            bool any_child_timer = false;
            for (const auto& node : this->marginal_nodes) {
                any_child_timer |= node->has_child_timer();

                const auto& metadata = node->get_metadata();
                const string& node_label = metadata->get_label();

                if (node->is_frozen()) {
                    // Observations were provided for this node so there's no
                    // need to sample from it. Its probability is deterministic.
                    Eigen::VectorXd probabilities =
                        Eigen::VectorXd::Zero(metadata->get_cardinality());
                    int value = node->get_assignment()(0, 0);
                    probabilities(value) = 1;
                    marginals.add_data(
                        node_label, Tensor3(probabilities), false);
                    continue;
                }

                if (time_step < metadata->get_initial_time_step()) {
                    Tensor3 prior(SamplerEstimator::get_prior(node));
                    marginals.add_data(metadata->get_label(), prior, false);
                    this->previous_marginals[node_label] =
                        marginals[node_label];

                    Tensor3 no_obs =
                        Tensor3::constant(1,
                                          this->num_particles,
                                          metadata->get_sample_size(),
                                          NO_OBS);
                    particles.add_data(node_label, no_obs);
                    continue;
                }

                if (sampled_particles.size() == 0 && !node->has_child_timer() &&
                    !any_child_timer) {
                    // No new observation. Just repeat the last marginal.
                    if (time_step == metadata->get_initial_time_step()) {
                        Tensor3 prior(SamplerEstimator::get_prior(node));
                        marginals.add_data(metadata->get_label(), prior, false);
                        this->previous_marginals[node_label] =
                            marginals[node_label];
                    }
                    else {
                        marginals.add_data(node_label,
                                           this->previous_marginals[node_label],
                                           false);
                    }
                    particles.add_data(node_label, node->get_assignment());
                    continue;
                }

                int children_template_time_step =
                    min(time_step, LAST_TEMPLATE_TIME_STEP);

                // p(child | node) from all child nodes that repeat over
                // time will be multiplied and accumulated in this variable.
                Eigen::MatrixXd repeatable_child_log_weights =
                    Eigen::MatrixXd::Zero(this->num_particles,
                                          metadata->get_cardinality());

                // p(child | node) from all child nodes that are not
                // repeatable will be multiplied and accumulated in this
                // variable.
                Eigen::MatrixXd single_time_child_log_weights =
                    Eigen::MatrixXd::Zero(this->num_particles,
                                          metadata->get_cardinality());

                // Posterior weights for child nodes that are timer are kept
                // separately because some of them will be accumulated (if
                // there's a jump) others won't.
                unordered_map<string, Eigen::MatrixXd>
                    segment_log_weights_per_timer;

                for (const auto& child :
                     this->get_marginal_node_children(node, time_step)) {

                    if (child->get_metadata()->is_timer()) {
                        const auto& child_timer =
                            dynamic_pointer_cast<TimerNode>(child);
                        const string& child_label =
                            child->get_metadata()->get_label();

                        segment_log_weights_per_timer[child_label] =
                            get_segment_log_weights(node, child_timer);

                        this->update_marginal_left_segment_distributions(
                            node, child_timer);
                    }
                    else {
                        Eigen::MatrixXd child_weights =
                            child->get_cpd()->get_posterior_weights(
                                child->get_parents(),
                                node,
                                child,
                                this->random_generators_per_job.size());

                        if (child->get_metadata()->is_replicable()) {
                            repeatable_child_log_weights.array() +=
                                (child_weights.array() + EPSILON).log();
                        }
                        else {
                            single_time_child_log_weights.array() +=
                                (child_weights.array() + EPSILON).log();
                        }
                    }
                }

                // Accumulate weights

                // Child weights from replicable nodes are always
                // accumulated because they come from samples that are
                // generated at every time step.
                this->cum_marginal_posterior_log_weights[node_label].array() +=
                    repeatable_child_log_weights.array();

                // We incorporate the most up to date weights from single
                // time children to compute the node's posterior.
                Eigen::MatrixXd weights_per_particle =
                    this->cum_marginal_posterior_log_weights[node_label]
                        .array() +
                    single_time_child_log_weights.array();

                // Now we process weights that come from timer nodes that
                // are children of the marginal node
                for (const auto& [timer_label, log_weights] :
                     segment_log_weights_per_timer) {
                    const auto& timer = dynamic_pointer_cast<TimerNode>(
                        this->template_dbn.get_node(
                            timer_label, children_template_time_step));

                    for (int i = 0; i < this->num_particles; i++) {
                        // We use the current state of the segment to update
                        // the posterior of the marginal node.
                        weights_per_particle.row(i).array() +=
                            log_weights.row(i).array();

                        if (timer->get_forward_assignment()(i, 0) == 0) {
                            // New segment. We can incorporate the weights
                            // related to the previous last segment to the
                            // cumulative weight as the duration of such
                            // segment does not change anymore and,
                            // therefore, p(timer | node) do not change for
                            // that segment.
                            this->cum_marginal_posterior_log_weights[node_label]
                                .row(i)
                                .array() += log_weights.row(i).array();
                        }
                    }
                }

                // Compute weights for the current posteriors per particle.
                // Open segments are considered here but they are only
                // accumulates when they become closed, which is when the
                // semi-Markov model jumps to a new sequence of states in a
                // different segment.
                weights_per_particle.colwise() -=
                    weights_per_particle.rowwise().maxCoeff();
                weights_per_particle = weights_per_particle.array().exp();
                Eigen::VectorXd sum_per_row =
                    weights_per_particle.rowwise().sum();
                weights_per_particle = weights_per_particle.array().colwise() /
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
                bool process_left_segment = EXISTS(
                    node_label,
                    this->last_left_segment_marginal_nodes_distribution_indices);

                int i = 0;
                for (const auto& posterior : posteriors) {
                    probabilities.array() += posterior->get_values(0).array();

                    // Each particle holds its own marginal node posterior
                    double curr_value = node->get_assignment()(i, 0);
                    double new_value = posterior->sample(
                        this->random_generators_per_job[0], 0)(0);

                    if (curr_value != new_value) {
                        node->set_assignment(i, 0, new_value);

                        // Update time controlled node's left segment
                        // distributions with the most up to date marginal
                        // node value. A marginal node has only a single
                        // time instance and, therefore, its current value
                        // must influence currently open segments.
                        if (time_step > 0 && process_left_segment) {
                            for (
                                const auto& [timer_label, log_weights] :
                                this->last_left_segment_marginal_nodes_distribution_indices
                                    [node_label]) {

                                const auto& timer =
                                    dynamic_pointer_cast<TimerNode>(
                                        this->template_dbn.get_node(
                                            timer_label,
                                            children_template_time_step));
                                const string& timed_node_label =
                                    timer->get_controlled_node()
                                        ->get_metadata()
                                        ->get_label();
                                auto& distribution_indices =
                                    this->last_left_segment_distribution_indices
                                        [timed_node_label];

                                int rcc = timer->get_cpd()
                                              ->get_parent_label_to_indexing()
                                              .at(node_label)
                                              .right_cumulative_cardinality;
                                distribution_indices(i) +=
                                    (new_value - curr_value) * rcc;
                            }
                        }
                    }
                    i++;
                }

                // Final estimate for a marginal node consider its posterior
                // in all the particles.
                probabilities.array() /= probabilities.sum();
                marginals.add_data(node_label, Tensor3(probabilities), false);
                this->previous_marginals[node_label] = marginals[node_label];

                // We also include the particles in case it's necessary
                // Save particles
                Tensor3 filtered_samples_tensor(node->get_assignment());
                if (node->get_assignment().cols() > 1) {
                    // If the sample has more than one dimension, we
                    // move the dimension to the depth axis of the
                    // tensor because the column is reserved for the
                    // time dimension.
                    filtered_samples_tensor.reshape(
                        node->get_assignment().cols(), this->num_particles, 1);
                }
                particles.add_data(node_label, filtered_samples_tensor);
            }

            return marginals;
        }

        Eigen::MatrixXd ParticleFilter::get_segment_log_weights(
            const RVNodePtr& parent_node,
            const TimerNodePtr& child_timer) const {
            Eigen::MatrixXd log_weights;

            const auto& parent_metadata = parent_node->get_metadata();
            const auto& timer_metadata = child_timer->get_metadata();
            if (child_timer->get_time_step() == 0) {
                // Uniform at the first time step because
                // p(d >= 0 | . ) = 1.
                log_weights = Eigen::MatrixXd::Zero(
                    this->num_particles, parent_metadata->get_cardinality());
            }
            else {
                // Compute posterior weight for the left segment
                // timer. There's no need to compute the
                // posterior weights for the central timer
                // because p(d >= 0 | . ) is always 1 regardless
                // of the value of the timer's parent node. This
                // happens with because there are no more segments
                // to the right as inference in this filter is
                // done one time slice at a time.
                const auto& left_segment_timer =
                    dynamic_pointer_cast<TimerNode>(
                        child_timer->get_previous());

                const auto& distribution_indices =
                    last_left_segment_marginal_nodes_distribution_indices
                        .at(parent_metadata->get_label())
                        .at(timer_metadata->get_label());

                Eigen::MatrixXd segment_weight =
                    left_segment_timer->get_cpd()
                        ->get_particle_timer_posterior_weights(
                            distribution_indices,
                            parent_node,
                            left_segment_timer,
                            this->random_generators_per_job.size());

                log_weights = (segment_weight.array() + EPSILON).log();
            }

            return log_weights;
        }

        void ParticleFilter::update_marginal_left_segment_distributions(
            const RVNodePtr& parent_node, const TimerNodePtr& child_timer) {

            // Get the list of distributions (one per particle)
            // when the parent node is 0. This is the baseline for us to get the
            // other distributions (one for each possible value the parent node
            // can take).
            Eigen::MatrixXd current_assignment = parent_node->get_assignment();
            parent_node->set_assignment(Eigen::MatrixXd::Zero(
                this->num_particles, current_assignment.cols()));
            Eigen::VectorXi new_distribution_indices =
                child_timer->get_cpd()->get_indexed_distribution_indices(
                    child_timer->get_parents(), this->num_particles);
            parent_node->set_assignment(current_assignment);

            const string& parent_label =
                parent_node->get_metadata()->get_label();
            const string& timer_label =
                child_timer->get_metadata()->get_label();

            if (EXISTS(parent_label,
                       last_left_segment_marginal_nodes_distribution_indices) &&
                EXISTS(timer_label,
                       last_left_segment_marginal_nodes_distribution_indices
                           [parent_label])) {

                auto& curr_distribution_indices =
                    last_left_segment_marginal_nodes_distribution_indices
                        [parent_label][timer_label];

                // Replace distribution whenever a new segment starts
                for (int i = 0; i < this->num_particles; i++) {
                    if (child_timer->get_forward_assignment()(i, 0) == 0) {
                        curr_distribution_indices(i) =
                            new_distribution_indices(i);
                    }
                }
            }
            else {
                last_left_segment_marginal_nodes_distribution_indices
                    [parent_label][timer_label] = new_distribution_indices;
            }
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
                RVNodePtrVec previous_nodes =
                    this->template_dbn.get_data_nodes(template_time_step - 1);
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
                    if (node->get_metadata()->is_timer()) {
                        last_particles.add_data(
                            node->get_metadata()->get_label(),
                            dynamic_pointer_cast<TimerNode>(node)
                                ->get_forward_assignment());
                    }
                    else {
                        last_particles.add_data(
                            node->get_metadata()->get_label(),
                            node->get_assignment());
                    }
                }
                EvidenceSet last_particles_previous_nodes;
                for (const auto& node : previous_nodes) {
                    if (node->get_metadata()->is_timer()) {
                        last_particles_previous_nodes.add_data(
                            node->get_metadata()->get_label(),
                            dynamic_pointer_cast<TimerNode>(node)
                                ->get_forward_assignment());
                    }
                    else {
                        last_particles_previous_nodes.add_data(
                            node->get_metadata()->get_label(),
                            node->get_assignment());
                    }
                }

                // Save accumulated weights and posteriors as well
                auto last_cum_marginal_posterior_log_weights =
                    this->cum_marginal_posterior_log_weights;
                auto last_last_left_segment_distribution_indices =
                    this->last_left_segment_distribution_indices;
                auto last_last_left_segment_marginal_nodes_distribution_indices =
                    this->last_left_segment_marginal_nodes_distribution_indices;

                EvidenceSet empty_set;
                int initial_time_step = this->last_time_step + 1;
                int final_time_step = this->last_time_step + num_time_steps;
                for (int t = initial_time_step; t <= final_time_step; t++) {
                    this->elapse(empty_set, t);
                    Eigen::VectorXi no_particles =
                        this->weigh_and_sample_particles(t, empty_set);
                    EvidenceSet resampled_particles =
                        this->resample(empty_set, t, no_particles);
                    this->apply_rao_blackwellization(
                        t, resampled_particles, no_particles);
                    particles.hstack(resampled_particles);
                    this->update_left_segment_distribution_indices(t);
                    this->move_particles_back_in_time(t);
                }

                // Restore particles
                for (const auto& node : nodes) {
                    auto samples =
                        last_particles[node->get_metadata()->get_label()](0, 0);
                    if (node->get_metadata()->is_timer()) {
                        dynamic_pointer_cast<TimerNode>(node)
                            ->set_forward_assignment(move(samples));
                    }
                    else {
                        node->set_assignment(move(samples));
                    }
                }
                for (const auto& node : previous_nodes) {
                    auto samples =
                        last_particles_previous_nodes[node->get_metadata()
                                                          ->get_label()](0, 0);
                    if (node->get_metadata()->is_timer()) {
                        dynamic_pointer_cast<TimerNode>(node)
                            ->set_forward_assignment(move(samples));
                    }
                    else {
                        node->set_assignment(move(samples));
                    }
                }

                // Restore weights and posteriors
                this->cum_marginal_posterior_log_weights =
                    last_cum_marginal_posterior_log_weights;
                this->last_left_segment_distribution_indices =
                    last_last_left_segment_distribution_indices;
                this->last_left_segment_marginal_nodes_distribution_indices =
                    last_last_left_segment_marginal_nodes_distribution_indices;
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

        Eigen::VectorXi
        ParticleFilter::shuffle_rows(const Eigen::VectorXi& original_vector,
                                     const Eigen::VectorXi& indices) const {
            Eigen::MatrixXd matrix(original_vector.size(), 1);
            matrix.col(0) = original_vector.cast<double>();
            matrix = this->shuffle_rows(matrix, indices);
            return matrix.col(0).cast<int>();
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
                partially_shuffled_matrix.row(i - initial_row) =
                    original_matrix.row(idx);
            }

            scoped_lock lock(shuffled_matrix_mutex);
            shuffled_matrix.block(
                initial_row, 0, num_rows, original_matrix.cols()) =
                partially_shuffled_matrix;
        }

        void ParticleFilter::update_left_segment_distribution_indices(
            int time_step) {
            int template_time_step = min(time_step, LAST_TEMPLATE_TIME_STEP);

            for (const auto& node_label : this->time_controlled_node_set) {
                RVNodePtr node =
                    this->template_dbn.get_node(node_label, template_time_step);

                // Store the distributions of the most recent left
                // segment
                auto timer = dynamic_pointer_cast<TimerNode>(node->get_timer());
                Eigen::VectorXi distribution_indices =
                    timer->get_cpd()->get_indexed_distribution_indices(
                        timer->get_parents(), this->num_particles);
                const string& timed_node_label =
                    timer->get_controlled_node()->get_metadata()->get_label();

                if (time_step == 0) {
                    this->last_left_segment_distribution_indices[node_label] =
                        distribution_indices;
                }
                else {
                    Eigen::VectorXi& curr_indices =
                        this->last_left_segment_distribution_indices
                            [node_label];

                    // Replace indices where a new segment started
                    for (int i = 0; i < this->num_particles; i++) {
                        if (timer->get_forward_assignment()(i, 0) == 0) {
                            curr_indices(i) = distribution_indices(i);
                        }
                    }
                }
            }
        }

        RVNodePtrVec ParticleFilter::get_marginal_node_children(
            const RVNodePtr& marginal_node, int time_step) const {

            int children_template_time_step =
                min(time_step, LAST_TEMPLATE_TIME_STEP);

            RVNodePtrVec children_at_time_step;

            if (marginal_node->has_child_at(children_template_time_step)) {
                children_at_time_step =
                    marginal_node->get_children(children_template_time_step);
            }
            const auto& single_time_children =
                marginal_node->get_single_time_children();
            children_at_time_step.insert(children_at_time_step.end(),
                                         single_time_children.begin(),
                                         single_time_children.end());

            return children_at_time_step;
        }

        //----------------------------------------------------------------------
        // Getters & Setters
        //----------------------------------------------------------------------

        void ParticleFilter::set_show_progress(bool show_progress) {
            this->show_progress = show_progress;
        }

    } // namespace model
} // namespace tomcat
