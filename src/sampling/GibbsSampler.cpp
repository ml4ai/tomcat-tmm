#include "GibbsSampler.h"

#include <thread>
#include <unordered_set>

// This is deprecated. The new version is in boost/timer/progress_display.hpp
// but only available for boost 1.72
#include <boost/progress.hpp>
#include <boost/range/adaptor/reversed.hpp>
#include <gsl/gsl_rng.h>

#include "pgm/TimerNode.h"
#include "utils/Multithreading.h"

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        GibbsSampler::GibbsSampler(const shared_ptr<DynamicBayesNet>& model,
                                   int burn_in_period,
                                   int num_jobs)
            : Sampler(model, num_jobs), burn_in_period(burn_in_period),
              keep_sample_mutex(make_unique<mutex>()) {}

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

        void GibbsSampler::print_nodes() const {
            //            Eigen::MatrixXd back_timers(3, 10);
            //            Eigen::MatrixXd for_timers(3, 10);
            //            Eigen::MatrixXd states(3, 10);
            //            for (int i = 0; i < 10; i++) {
            //                back_timers.col(i) =
            //                    this->model->get_node("Timer",
            //                    i)->get_assignment().col(0);
            //                for_timers.col(i) =
            //                dynamic_pointer_cast<TimerNode>(
            //                                        this->model->get_node("Timer",
            //                                        i))
            //                                        ->get_forward_assignment()
            //                                        .col(0);
            //                states.col(i) =
            //                    this->model->get_node("State",
            //                    i)->get_assignment().col(0);
            //            }
            //
            //            cout << "STATES" << endl;
            //            cout << "-----------------" << endl;
            //            cout << states << endl << endl;
            //
            //            cout << "F-TIMERS" << endl;
            //            cout << "-----------------" << endl;
            //            cout << for_timers << endl << endl;
            //
            //            cout << "B-TIMERS" << endl;
            //            cout << "-----------------" << endl;
            //            cout << back_timers << endl << endl;

            for (const auto& node : this->node_set.sampled_nodes) {
                cout << node->get_timed_name() << endl;
                cout << "-----------------" << endl;
                cout << node->get_assignment() << endl << endl;
            }
        }

        void
        GibbsSampler::sample_latent(const shared_ptr<gsl_rng>& random_generator,
                                    int num_samples) {
            this->reset();
            vector<shared_ptr<gsl_rng>> random_generators_per_job =
                split_random_generator(random_generator, this->num_jobs);
            this->fill_initial_samples(random_generator);
            this->init_samples_storage(num_samples,
                                       this->node_set.sampled_nodes);
            this->init_timers(this->node_set.timer_nodes);

            bool discard = true;
            LOG("Burn-in");
            boost::progress_display progress(this->burn_in_period);

            // Gibbs step
            for (int i = 0; i < this->burn_in_period + num_samples; i++) {
                if (i >= burn_in_period && discard) {
                    discard = false;
                    LOG("Sampling");
                    progress.restart(num_samples);
                }

                this->sample_nodes_in_parallel(
                    random_generators_per_job,
                    this->node_set.even_time_data_nodes_per_job,
                    discard,
                    true);

                this->sample_nodes_in_parallel(
                    random_generators_per_job,
                    this->node_set.odd_time_data_nodes_per_job,
                    discard,
                    true);

                this->sample_single_thread_nodes(
                    random_generators_per_job,
                    this->node_set.single_thread_over_time_nodes,
                    discard);

                this->update_timer_sufficient_statistics(
                    this->node_set.timer_nodes);

                this->sample_nodes_in_parallel(
                    random_generators_per_job,
                    this->node_set.parameter_nodes_per_job,
                    discard,
                    false);

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

        GibbsSampler::NodeSet GibbsSampler::get_node_set() const {
            NodeSet node_set;

            int num_params = this->model->get_parameter_nodes().size();
            node_set.parameter_nodes_per_job = vector<vector<shared_ptr<Node>>>(
                min(this->num_jobs, num_params));
            node_set.even_time_data_nodes_per_job =
                vector<vector<shared_ptr<Node>>>(this->num_jobs);
            node_set.odd_time_data_nodes_per_job =
                vector<vector<shared_ptr<Node>>>(this->num_jobs);
            int time_steps = this->model->get_time_steps();
            int time_steps_per_job =
                ceil(time_steps / (double)(2 * this->num_jobs));

            // We proceed from the root to the leaves so that child nodes
            // can update the sufficient statistics of parameter nodes correctly
            // given that parent nodes were already sampled.
            int params = 0;
            for (auto& node : this->model->get_nodes_topological_order()) {
                shared_ptr<RandomVariableNode> rv_node =
                    dynamic_pointer_cast<RandomVariableNode>(node);

                // We only generate samples for nodes within the time range
                // set.
                int t = rv_node->get_time_step();
                int t_min = this->min_time_step_to_sample;
                int t_max = this->max_time_step_to_sample >= 0
                                ? this->max_time_step_to_sample
                                : this->model->get_time_steps() - 1;
                //                if (t < t_min || t > t_max) {
                //                    continue;
                //                }
                if (t < t_min) {
                    continue;
                }

                if (!this->sample_after_max_time_step && t > t_max) {
                    continue;
                }

                if (node->get_metadata()->is_parameter()) {
                    // We don't sample parameter nodes that are frozen or if
                    // the sampler is not trainable
                    if (rv_node->is_frozen() || !this->trainable) {
                        continue;
                    }

                    int job = params % this->num_jobs;
                    params++;
                    node_set.parameter_nodes_per_job[job].push_back(node);
                    node_set.sampled_nodes.push_back(node);
                }
                else {
                    // We don't check if a data node is frozen to insert it
                    // on the list because even the frozen ones (observable
                    // nodes) need to be processed to update the sufficient
                    // statistics of the parameter nodes their CPDs depend on.
                    node_set.sampled_nodes.push_back(node);

                    // Timer nodes and the nodes controlled by them cannot be
                    // processed in parallel as the size of the left and
                    // segments to which the controlled nodes belong to in
                    // random.

                    if (node->get_metadata()->is_timer()) {
                        node_set.timer_nodes.push_back(node);
                        node_set.single_thread_over_time_nodes.push_back(node);
                    }
                    else {
                        bool multiple_connections_over_time =
                            !node->get_metadata()->is_replicable() &&
                            !node->get_metadata()->is_single_time_link();
                        const auto& rv_node =
                            dynamic_pointer_cast<RandomVariableNode>(node);
                        if (multiple_connections_over_time ||
                            rv_node->has_timer() || t > t_max) {
                            // The nodes after t_max must be sampled from
                            // their priors, and therefore, they have to be
                            // sampled sequentially.
                            node_set.single_thread_over_time_nodes.push_back(
                                node);
                        }
                        else {
                            int job = 0;
                            int time_step = rv_node->get_time_step();
                            if (time_step % 2 == 0) {
                                job = time_step / (2 * time_steps_per_job);
                                node_set.even_time_data_nodes_per_job[job]
                                    .push_back(node);
                            }
                            else {
                                job =
                                    (time_step - 1) / (2 * time_steps_per_job);
                                node_set.odd_time_data_nodes_per_job[job]
                                    .push_back(node);
                            }
                        }
                    }
                }
            }

            return node_set;
        }

        void GibbsSampler::init_timers(
            const std::vector<std::shared_ptr<Node>> timer_nodes) {
            // The ancestral sampling will fill the duration of a segment
            // first, which means that the timer counter will be
            // filled backwards (d, d-1, d-2... instead of 0, 1, 2, ...).
            // Therefore we sample the timer nodes from their posterior to fix
            // the counters.
            for (auto& timer_node : timer_nodes) {
                this->sample_from_posterior({nullptr}, timer_node, true, false);
            }
        }

        void GibbsSampler::fill_initial_samples(
            const shared_ptr<gsl_rng>& random_generator) {
            // Observable nodes are already frozen by the Gibbs sampler thus
            // there's no need to add them as data in the ancestral sampler.
            AncestralSampler initial_sampler(this->model);
            initial_sampler.set_min_time_step_to_sample(
                this->min_initialization_time_step);
            initial_sampler.set_max_time_step_to_sample(
                this->max_time_step_to_sample);
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

        void GibbsSampler::sample_nodes_in_parallel(
            const std::vector<std::shared_ptr<gsl_rng>>&
                random_generators_per_job,
            const std::vector<std::vector<std::shared_ptr<Node>>>&
                nodes_per_job,
            bool discard,
            bool data_nodes) {

            if (this->num_jobs == 1) {
                // Avoid the overhead of creating a new thread and use the main
                // one.
                this->run_sample_from_posterior_thread(
                    random_generators_per_job.at(0),
                    nodes_per_job.at(0),
                    discard);
            }
            else {
                vector<thread> node_threads;
                for (int job = 0; job < this->num_jobs; job++) {
                    thread node_thread(
                        &GibbsSampler::run_sample_from_posterior_thread,
                        this,
                        random_generators_per_job.at(job),
                        nodes_per_job.at(job),
                        discard);
                    node_threads.push_back(move(node_thread));
                }

                for (auto& node_thread : node_threads) {
                    node_thread.join();
                }
            }
        }

        void GibbsSampler::run_sample_from_posterior_thread(
            const std::shared_ptr<gsl_rng>& random_generator,
            const std::vector<std::shared_ptr<Node>>& nodes,
            bool discard) {

            for (auto& node : nodes) {
                this->sample_from_posterior({random_generator}, node, discard);
            }
        }

        void GibbsSampler::sample_from_posterior(
            const vector<shared_ptr<gsl_rng>>& random_generator_per_job,
            const shared_ptr<Node>& node,
            bool discard,
            bool update_sufficient_statistics) {

            shared_ptr<RandomVariableNode> rv_node =
                dynamic_pointer_cast<RandomVariableNode>(node);

            if (!rv_node->is_frozen()) {
                // We need to pass the max time step to sample so the rv node
                // ignores the children at a future time step when computing
                // the posterior weights for the node.
                int max_time_step = this->max_time_step_to_sample >= 0
                                        ? this->max_time_step_to_sample
                                        : this->model->get_time_steps() - 1;
                Eigen::MatrixXd sample(0, 0);
                if (rv_node->get_time_step() > max_time_step) {
                    sample = rv_node->sample(random_generator_per_job,
                                             this->num_in_plate_samples);
                }
                else {
                    sample = rv_node->sample_from_posterior(
                        random_generator_per_job, max_time_step);
                }

                if (rv_node->get_metadata()->is_timer()) {
                    dynamic_pointer_cast<TimerNode>(rv_node)
                        ->set_forward_assignment(sample);
                }
                else {
                    rv_node->set_assignment(sample);
                }
                if (!discard) {
                    this->keep_sample(rv_node, sample);
                }
            }

            if (this->trainable) {
                if (node->get_metadata()->is_parameter()) {
                    // As nodes are processed, the sufficient statistic
                    // table of their dependent parent parameter nodes are
                    // updated accordingly. So at this point we already have all
                    // the information needed to sample the parameter from its
                    // posterior.
                    rv_node->reset_sufficient_statistics();
                }
                else if (update_sufficient_statistics) {
                    rv_node->update_parents_sufficient_statistics();
                }
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

        void GibbsSampler::sample_single_thread_nodes(
            const vector<shared_ptr<gsl_rng>>& random_generator_per_job,
            const vector<shared_ptr<Node>>& single_thread_nodes,
            bool discard) {
            for (auto& node : single_thread_nodes) {
                // We only update the sufficient statistics for timer
                // nodes after, when all the time controlled nodes have
                // been sampled.
                bool upd_suff_stat = !node->get_metadata()->is_timer();
                this->sample_from_posterior(
                    random_generator_per_job, node, discard, upd_suff_stat);
            }
        }

        void GibbsSampler::update_timer_sufficient_statistics(
            const std::vector<std::shared_ptr<Node>>& timer_nodes) {
            // Update sufficient statistics for timer nodes and fill
            // backwards counters
            for (int j = 0; j < timer_nodes.size(); j++) {
                if (j < timer_nodes.size() - 1) {
                    // We do not add the last timer to the sufficient
                    // statistics of the duration's prior distribution
                    // to avoid biasing it with durations of truncated
                    // segments.
                    const auto& timer =
                        dynamic_pointer_cast<TimerNode>(timer_nodes.at(j));
                    timer->update_parents_sufficient_statistics();
                }

                const auto& reverse_timer = dynamic_pointer_cast<TimerNode>(
                    timer_nodes.at(timer_nodes.size() - j - 1));
                int max_time_step = this->max_time_step_to_sample >= 0
                                        ? this->max_time_step_to_sample
                                        : this->model->get_time_steps();
                reverse_timer->update_backward_assignment(max_time_step);
            }
        }

        Tensor3 GibbsSampler::get_samples(const string& node_label) const {
            if (!EXISTS(node_label, this->node_label_to_samples)) {
                stringstream ss;
                ss << "The node " << node_label
                   << " does not belong to the "
                      "model.";
                throw invalid_argument(ss.str());
            }

            return this->node_label_to_samples.at(node_label);
        }

        Tensor3 GibbsSampler::get_samples(const std::string& node_label,
                                          int low_time_step,
                                          int high_time_step) const {

            if (!EXISTS(node_label, this->node_label_to_samples)) {
                stringstream ss;
                ss << "The node " << node_label
                   << " does not belong to the "
                      "model.";
                throw invalid_argument(ss.str());
            }

            return this->node_label_to_samples.at(node_label)
                .slice(low_time_step, high_time_step, 2);
        }

        void GibbsSampler::get_info(nlohmann::json& json) const {
            json["name"] = "gibbs";
            json["burn_in"] = this->burn_in_period;
        }

        unique_ptr<Sampler> GibbsSampler::clone() const {
            unique_ptr<Sampler> new_sampler = make_unique<GibbsSampler>(
                this->model, this->burn_in_period, this->num_jobs);
            // Clone the model and the nodes in it
            new_sampler->set_model(
                make_shared<DynamicBayesNet>(this->model->clone(true)));

            return new_sampler;
        }

        unordered_set<string> GibbsSampler::get_sampled_node_labels() const {
            unordered_set<string> labels;
            labels.reserve(this->node_label_to_samples.size());

            for (auto key_value : this->node_label_to_samples) {
                labels.insert(key_value.first);
            }

            return labels;
        }

        void GibbsSampler::prepare() { this->node_set = this->get_node_set(); }

    } // namespace model
} // namespace tomcat
