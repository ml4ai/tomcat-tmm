#include "GibbsSampler.h"

#include <thread>
#include <unordered_set>

// This is deprecated. The new version is in boost/timer/progress_display.hpp
// but only available for boost 1.72
#include <boost/progress.hpp>
#include <boost/range/adaptor/reversed.hpp>
#include <gsl/gsl_rng.h>

#include "AncestralSampler.h"
#include "pgm/TimerNode.h"

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        GibbsSampler::GibbsSampler(const shared_ptr<DynamicBayesNet>& model,
                                   int burn_in_period,
                                   int num_jobs)
            : Sampler(model), burn_in_period(burn_in_period),
              num_jobs(num_jobs) {
            this->keep_sample_mutex = make_unique<mutex>();
        }

        GibbsSampler::~GibbsSampler() {}

        //----------------------------------------------------------------------
        // Copy & Move constructors/assignments
        //----------------------------------------------------------------------
        GibbsSampler& GibbsSampler::operator=(const GibbsSampler& sampler) {
            this->copy_sampler(sampler);
            return *this;
        }

        GibbsSampler::GibbsSampler(const GibbsSampler& sampler) {
            this->copy_sampler(sampler);
        }

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        void GibbsSampler::copy_sampler(const GibbsSampler& sampler) {
            Sampler::copy_sampler(sampler);
            this->burn_in_period = sampler.burn_in_period;
            this->node_label_to_samples = sampler.node_label_to_samples;
            this->iteration = sampler.iteration;
            this->keep_sample_mutex = make_unique<mutex>();
        }

        void
        GibbsSampler::sample_latent(const shared_ptr<gsl_rng>& random_generator,
                                    int num_samples) {
            this->reset();
            // List of all nodes sampled: parameter and data nodes.
            vector<shared_ptr<Node>> sampled_nodes;
            vector<shared_ptr<Node>> parameter_nodes;
            vector<shared_ptr<Node>> timer_nodes;
            vector<shared_ptr<Node>> timer_controlled_nodes;

            // We divide data nodes in two categories: data nodes at even and
            // odd time steps. This guarantees that nodes in each of these
            // categories can be sampled in parallel as nodes from a time
            // step can only depend on nodes from previous, current
            // or subsequent time steps.
            vector<vector<shared_ptr<Node>>> even_time_data_nodes_per_job(
                this->num_jobs);
            vector<vector<shared_ptr<Node>>> odd_time_data_nodes_per_job(
                this->num_jobs);
            int time_steps = this->model->get_time_steps();
            int time_steps_per_job =
                ceil(time_steps / (double)(2 * this->num_jobs));

            // We proceed from the root to the leaves so that child nodes
            // can update the sufficient statistics of parameter nodes correctly
            // given that parent nodes were already sampled.
            for (auto& node : this->model->get_nodes_topological_order()) {
                if (node->get_metadata()->is_parameter()) {
                    shared_ptr<RandomVariableNode> rv_node =
                        dynamic_pointer_cast<RandomVariableNode>(node);
                    if (!rv_node->is_frozen()) {
                        if (this->max_time_step_to_sample < 0 ||
                            rv_node->get_time_step() <=
                                this->max_time_step_to_sample) {
                            // TODO - change this if we need to have a parameter
                            //  that depends on other parameters.
                            parameter_nodes.push_back(node);
                            sampled_nodes.push_back(node);
                        }
                    }
                }
                else {
                    // We don't check if a data node is frozen to insert it
                    // on the list because even the frozen ones (observable
                    // nodes) need to be processed to update the sufficient
                    // statistics of the parameter nodes their CPDs depend on.
                    sampled_nodes.push_back(node);

                    // Timer nodes and the nodes controlled by them cannot be
                    // processed in parallel as the size of the left and
                    // segments to which the controlled nodes belong to in
                    // random.
                    if (node->get_metadata()->is_timer()) {
                        timer_nodes.push_back(node);
                    }
                    else {
                        const auto& rv_node =
                            dynamic_pointer_cast<RandomVariableNode>(node);
                        if (rv_node->has_timer()) {
                            timer_controlled_nodes.push_back(node);
                        }
                        else {
                            int job = 0;
                            int time_step = rv_node->get_time_step();
                            if (time_step % 2 == 0) {
                                job = time_step / (2 * time_steps_per_job);
                                even_time_data_nodes_per_job[job].push_back(
                                    node);
                            }
                            else {
                                job =
                                    (time_step - 1) / (2 * time_steps_per_job);
                                odd_time_data_nodes_per_job[job].push_back(
                                    node);
                            }
                        }
                    }
                }
            }

            this->fill_initial_samples(random_generator);
            this->init_samples_storage(num_samples, sampled_nodes);
            bool discard = true;
            LOG("Burn-in");
            boost::progress_display progress(this->burn_in_period);

            vector<shared_ptr<gsl_rng>> random_generators_per_job;
            if (this->num_jobs == 1) {
                // Use the main thread random generator
                random_generators_per_job.push_back(random_generator);
            }
            else {
                // Use the main thread random generator to generate individual
                // random number generators with different seeds per thread
                for (int job = 0; job < this->num_jobs; job++) {
                    shared_ptr<gsl_rng> gen(gsl_rng_alloc(gsl_rng_mt19937));
                    long int seed = gsl_rng_uniform_int(gen.get(), 1000);
                    gsl_rng_set(gen.get(), seed);
                    random_generators_per_job.push_back(move(gen));
                }
            }

            // The ancestral sampling will fill the duration of a segment
            // first, which means that the timer counter will be
            // filled backwards (d, d-1, d-2... instead of 0, 1, 2, ...).
            // Therefore we sample the timer nodes from their posterior to fix
            // the counters.
            this->update_timer_nodes(timer_nodes, true, false);

            // Gibbs step
            for (int i = 0; i < this->burn_in_period + num_samples; i++) {
                if (i >= burn_in_period && discard) {
                    discard = false;
                    LOG("Sampling");
                    progress.restart(num_samples);
                }

                this->sample_data_nodes_in_parallel(
                    random_generators_per_job,
                    even_time_data_nodes_per_job,
                    discard);

                this->sample_data_nodes_in_parallel(random_generators_per_job,
                                                    odd_time_data_nodes_per_job,
                                                    discard);

                for (auto& node : timer_controlled_nodes) {
                    this->sample_data_node(random_generator, node, discard);
                }

                this->update_timer_nodes(timer_nodes, discard);

                for (auto& node : parameter_nodes) {
                    this->sample_parameter_node(
                        random_generator, node, discard);
                }

                ++progress;
                if (!discard) {
                    this->iteration++;
                }
            }
        }

        void GibbsSampler::reset() {
            this->iteration = 0;
            this->node_label_to_samples.clear();
        }

        void GibbsSampler::fill_initial_samples(
            const shared_ptr<gsl_rng>& random_generator) {
            // Observable nodes are already frozen by the Gibbs sampler thus
            // there's no need to add them as data in the ancestral sampler.
            AncestralSampler initial_sampler(this->model);
            initial_sampler.set_num_in_plate_samples(
                this->num_in_plate_samples);
            initial_sampler.sample(random_generator, 1);
        }

        void GibbsSampler::init_samples_storage(
            int num_samples, const vector<shared_ptr<Node>>& latent_nodes) {
            // If there's no observation for a node in a specific time step,
            // this might be inferred by the value in the column that
            // represent such time step in the matrix of samples as it's
            // going to be filled with the original value = -1. Therefore,
            // all the matrices of samples have the same size, regardless of
            // the node's initial time step.
            for (const auto& node : latent_nodes) {
                string node_label = node->get_metadata()->get_label();
                if (!EXISTS(node_label, this->node_label_to_samples)) {
                    int sample_size = node->get_metadata()->get_sample_size();
                    this->node_label_to_samples[node_label] =
                        Tensor3::constant(sample_size,
                                          num_samples,
                                          this->model->get_time_steps(),
                                          -1);
                }
            }
        }

        void GibbsSampler::sample_data_nodes_in_parallel(
            const std::vector<std::shared_ptr<gsl_rng>>&
                random_generators_per_job,
            const std::vector<std::vector<std::shared_ptr<Node>>>&
                nodes_per_job,
            bool discard) {

            if (this->num_jobs == 1) {
                // Avoid overhead of creating a new thread and use the main one.
                this->run_data_node_thread(random_generators_per_job.at(0),
                                           nodes_per_job.at(0),
                                           discard);
            }
            else {
                vector<thread> data_node_threads;
                for (int job = 0; job < this->num_jobs; job++) {
                    thread data_node_thread(&GibbsSampler::run_data_node_thread,
                                            this,
                                            random_generators_per_job.at(job),
                                            nodes_per_job.at(job),
                                            discard);
                    data_node_threads.push_back(move(data_node_thread));
                }

                for (auto& data_node_thread : data_node_threads) {
                    data_node_thread.join();
                }
            }
        }

        void GibbsSampler::run_data_node_thread(
            const std::shared_ptr<gsl_rng>& random_generator,
            const std::vector<std::shared_ptr<Node>>& nodes,
            bool discard) {

            for (auto& node : nodes) {
                this->sample_data_node(random_generator, node, discard);
            }
        }

        void GibbsSampler::sample_data_node(
            const shared_ptr<gsl_rng>& random_generator,
            const shared_ptr<Node>& node,
            bool discard,
            bool update_sufficient_statistics) {

            shared_ptr<RandomVariableNode> rv_node =
                dynamic_pointer_cast<RandomVariableNode>(node);

            if (!rv_node->is_frozen()) {
                Eigen::MatrixXd sample =
                    rv_node->sample_from_posterior(random_generator);

                rv_node->set_assignment(sample);
                if (!discard) {
                    this->keep_sample(rv_node, sample);
                }
            }

            if (update_sufficient_statistics) {
                rv_node->update_parents_sufficient_statistics();
            }
        }

        void
        GibbsSampler::keep_sample(const shared_ptr<RandomVariableNode>& node,
                                  const Eigen::MatrixXd& sample) {
            if (sample.rows() == 1) {
                scoped_lock lock(*this->keep_sample_mutex);

                string node_label = node->get_metadata()->get_label();
                this->sampled_node_labels.insert(node_label);
                int time_step = node->get_time_step();
                for (int i = 0; i < sample.cols(); i++) {
                    this->node_label_to_samples.at(node_label)(
                        i, this->iteration, time_step) = sample(0, i);
                }
            }
        }

        void GibbsSampler::update_timer_nodes(
            const vector<shared_ptr<Node>>& timer_nodes,
            bool discard,
            bool update_sufficient_statistics) {

            for (auto& node : timer_nodes) {
                this->sample_data_node(
                    nullptr, node, discard, update_sufficient_statistics);
            }

            for (auto& node : boost::adaptors::reverse(timer_nodes)) {
                dynamic_pointer_cast<TimerNode>(node)
                    ->update_backward_assignment();
            }
        }

        void GibbsSampler::sample_parameter_node(
            const shared_ptr<gsl_rng>& random_generator,
            const shared_ptr<Node>& node,
            bool discard) {

            shared_ptr<RandomVariableNode> rv_node =
                dynamic_pointer_cast<RandomVariableNode>(node);
            Eigen::MatrixXd sample = rv_node->sample_from_conjugacy(
                random_generator, rv_node->get_size());
            rv_node->set_assignment(sample);

            // As nodes are processed, the sufficient statistic
            // table of their dependent parent parameter nodes are
            // updated accordingly. So at this point we already have all
            // the information needed to sample the parameter from its
            // posterior.
            rv_node->reset_sufficient_statistics();

            if (!discard) {
                this->keep_sample(rv_node, sample);
            }
        }

        Tensor3 GibbsSampler::get_samples(const string& node_label) const {
            return this->node_label_to_samples.at(node_label);
        }

        void GibbsSampler::get_info(nlohmann::json& json) const {
            json["name"] = "gibbs";
            json["burn_in"] = this->burn_in_period;
        }

    } // namespace model
} // namespace tomcat
