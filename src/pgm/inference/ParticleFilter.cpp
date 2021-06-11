#include "ParticleFilter.h"

#include <boost/progress.hpp>

#include "distribution/Categorical.h"
#include "distribution/Distribution.h"
#include "utils/Multithreading.h"

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
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

        void ParticleFilter::prepare() {
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

                    if (!node->get_metadata()->is_replicable()) {
                        // Each particle starts with the prior distribution
                        for (int i = 0; i < this->num_particles; i++) {
                            DistributionPtr distribution =
                                move(rv_node->get_cpd()
                                         ->get_distributions()[0]
                                         ->clone());
                            this->single_time_node_distribution_per_particle
                                [node->get_metadata()->get_label()]
                                    .push_back(distribution);
                        }

                        rv_node->set_assignment(
                            rv_node->get_cpd()
                                ->get_distributions()[0]
                                ->sample_many(this->random_generators_per_job,
                                              this->num_particles,
                                              0));
                    }
                }
            }
        }

        void ParticleFilter::create_template_dbn() {
            this->template_dbn = this->original_dbn.clone(false);
            this->template_dbn.unroll(LAST_TEMPLATE_TIME_STEP + 1, true);
        }

        EvidenceSet
        ParticleFilter::generate_particles(const EvidenceSet& new_data) {
            unique_ptr<boost::progress_display> progress;

            if (this->show_progress) {
                progress = make_unique<boost::progress_display>(
                    new_data.get_time_steps());
            }

            EvidenceSet particles;
            int initial_time_step = this->last_time_step + 1;
            int final_time_step =
                this->last_time_step + new_data.get_time_steps();
            for (int t = initial_time_step; t <= final_time_step; t++) {
                this->elapse(new_data, t);
                particles.hstack(this->resample(new_data, t));

                if (this->show_progress) {
                    ++(*progress);
                }
            }

            this->last_time_step = final_time_step;

            return particles;
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
                    if (node->get_metadata()->is_replicable()) {
                        // Single time nodes will be sampled directly from their
                        // updated posterior
                        Eigen::MatrixXd samples =
                            node->sample(this->random_generators_per_job,
                                         this->num_particles);
                        node->set_assignment(samples);
                    }
                }
            }
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
                }
            }

            EvidenceSet particles;
            for (const auto& node_label : this->data_node_labels) {
                particles.add_data(
                    node_label,
                    Tensor3::constant(1, this->num_particles, 1, NO_OBS));
            }

            RVNodePtrVec nodes =
                this->template_dbn.get_data_nodes(template_time_step);
            for (const auto& node : nodes) {
                if (!node->get_metadata()->is_replicable()) {
                    // Single time nodes will be sampled from their updated
                    // posterior next
                    continue;
                }

                const string& node_label = node->get_metadata()->get_label();

                const Eigen::MatrixXd& samples = node->get_assignment();
                Eigen::MatrixXd filtered_samples(this->num_particles,
                                                 samples.cols());
                if (node->is_frozen() || sampled_particles.size() == 0) {
                    filtered_samples = samples;
                }
                else {
                    for (int i = 0; i < this->num_particles; i++) {
                        int particle = sampled_particles(i, 0);
                        filtered_samples.row(i) = samples.row(particle);
                    }
                }

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

                if (time_step > LAST_TEMPLATE_TIME_STEP) {
                    const auto& prev_node = this->template_dbn.get_node(
                        node_label, LAST_TEMPLATE_TIME_STEP - 1);
                    // We save the assignment in the node at time step 1
                    // because next time we elapse, nodes in time step 2
                    // will be sampled. Therefore, the samples generated
                    // here will be assigned to the proper parent of the
                    // nodes at time step 2.
                    if (prev_node) {
                        prev_node->set_assignment(filtered_samples);
                    }
                }
                else {
                    node->set_assignment(filtered_samples);
                }
            }

            this->sample_single_time_nodes(particles, time_step);

            return particles;
        }

        void ParticleFilter::sample_single_time_nodes(EvidenceSet& particles,
                                                      int time_step) {
            for (const auto& node :
                 this->template_dbn.get_single_time_nodes()) {
                if (node->is_frozen()) {
                    // Observations were provided for this node so there's no
                    // need to sample from it.
                    continue;
                }

                const auto& metadata = node->get_metadata();
                if (metadata->get_initial_time_step() <= time_step) {
                    Eigen::MatrixXd log_weights = Eigen::MatrixXd::Zero(
                        this->num_particles,
                        node->get_metadata()->get_cardinality());

                    int template_time_step =
                        min(time_step, LAST_TEMPLATE_TIME_STEP);

                    for (const auto& child :
                         node->get_children(template_time_step)) {
                        Eigen::MatrixXd child_weights =
                            child->get_cpd()->get_posterior_weights(
                                child->get_parents(),
                                node,
                                child,
                                this->random_generators_per_job.size());

                        log_weights = (child_weights.array() + EPSILON).log();
                    }

                    // Unlog and normalize the weights
                    log_weights.colwise() -= log_weights.rowwise().maxCoeff();
                    log_weights = log_weights.array().exp();
                    Eigen::VectorXd sum_per_row = log_weights.rowwise().sum();
                    Eigen::MatrixXd posterior_weights =
                        (log_weights.array().colwise() / sum_per_row.array());

                    Eigen::MatrixXd samples(this->num_particles, 1);
                    for (int i = 0; i < this->num_particles; i++) {
                        auto& distribution =
                            this->single_time_node_distribution_per_particle
                                [metadata->get_label()][i];
                        distribution->update_from_posterior(
                            posterior_weights.row(i));
                        auto sample = distribution->sample(
                            this->random_generators_per_job[0], 0);
                        samples.conservativeResize(this->num_particles,
                                                   sample.size());
                        samples.row(i) = move(sample);
                    }

                    particles.add_data(metadata->get_label(), Tensor3(samples));
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

        //----------------------------------------------------------------------
        // Getters & Setters
        //----------------------------------------------------------------------

        void ParticleFilter::set_show_progress(bool show_progress) {
            this->show_progress = show_progress;
        }

    } // namespace model
} // namespace tomcat
