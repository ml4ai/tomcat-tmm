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
            this->step_counter = sampler.step_counter;
            this->keep_sample_mutex = make_unique<mutex>();
        }

        void GibbsSampler::print_nodes(const NodeSet& node_set) const {
            for (const auto& label : node_set.sampled_node_labels) {
                cout << label << endl;
                cout << "-----------------" << endl;
                cout << this->get_samples(label) << endl;
            }
        }

        void GibbsSampler::sample_latent(
            const shared_ptr<gsl_rng>& random_generator) {

            this->step_counter = 0;
            NodeSet node_set = this->get_node_set();
            vector<shared_ptr<gsl_rng>> random_generators_per_job =
                split_random_generator(random_generator, this->num_jobs);
            this->fill_initial_samples(random_generator);
            this->init_samples_storage(node_set.sampled_node_labels);
            this->init_timers(node_set.timer_nodes);

            unique_ptr<boost::progress_display> progress;
            if (this->show_progress) {
                cout << "Burn-in\n";
                progress =
                    make_unique<boost::progress_display>(this->burn_in_period);
            }

            // Gibbs step
            for (this->step_counter = 0;
                 this->step_counter < this->burn_in_period + this->num_samples;
                 this->step_counter++) {
                if (this->step_counter == burn_in_period &&
                    this->show_progress) {
                    cout << "Sampling\n";
                    progress->restart(this->num_samples);
                }

                this->sample_parallel_time_nodes(
                    random_generators_per_job,
                    node_set.even_time_data_nodes_per_job);

                this->sample_parallel_time_nodes(
                    random_generators_per_job,
                    node_set.odd_time_data_nodes_per_job);

                this->sample_sequential_time_nodes(
                    random_generators_per_job,
                    node_set.nodes_sampled_in_sequence);

                this->update_timer_backward_assignment(node_set.timer_nodes);

                this->update_timer_sufficient_statistics(node_set.timer_nodes);

                this->sample_parallel_time_nodes(
                    random_generators_per_job,
                    node_set.parameter_nodes_per_job);

                if (this->show_progress) {
                    ++(*progress);
                }
            }
        }

        GibbsSampler::NodeSet GibbsSampler::get_node_set() {
            NodeSet node_set;

            // We proceed from the root to the leaves so that child nodes
            // can update the sufficient statistics of parameter nodes
            // correctly given that parent nodes were already sampled.
            int params = 0;
            for (int t = 0; t < this->model->get_time_steps(); t++) {
                for (auto& node :
                     this->model->get_nodes_in_topological_order_at(t)) {

                    const string node_label = node->get_metadata()->get_label();
                    node_set.sampled_node_labels.insert(node_label);

                    if (node->get_metadata()->is_parameter()) {
                        if (node->is_frozen()) {
                            // We include frozen data nodes because they
                            // contribute to the sufficient statistics of the
                            // parameters' distributions. Frozen parameters do
                            // not need to be sampled.
                            continue;
                        }

                        int job = params % this->num_jobs;
                        params++;
                        if (job >= node_set.parameter_nodes_per_job.size()) {
                            node_set.parameter_nodes_per_job.resize(job + 1);
                        }
                        node_set.parameter_nodes_per_job[job].push_back(node);
                    }
                    else {
                        // Timer nodes and the nodes controlled by them cannot
                        // be processed in parallel as the size of the left and
                        // right segments to which the controlled nodes belong
                        // to are random.
                        if (node->get_metadata()->is_timer()) {
                            node_set.timer_nodes.push_back(node);
                            node_set.nodes_sampled_in_sequence.push_back(node);
                        }
                        else {
                            if (node->get_metadata()->is_multitime() ||
                                node->has_timer() || node->has_child_timer()) {
                                node_set.nodes_sampled_in_sequence.push_back(
                                    node);
                            }
                            else {
                                // Markov blankets
                                int time_steps = this->model->get_time_steps();
                                int time_steps_per_job = ceil(
                                    time_steps / (double)(2 * this->num_jobs));

                                int job = 0;
                                int time_step = node->get_time_step();
                                if (time_step % 2 == 0) {
                                    job = time_step / (2 * time_steps_per_job);
                                    if (job >=
                                        node_set.even_time_data_nodes_per_job
                                            .size()) {
                                        node_set.even_time_data_nodes_per_job
                                            .resize(job + 1);
                                    }
                                    node_set.even_time_data_nodes_per_job[job]
                                        .push_back(node);
                                }
                                else {
                                    job = (time_step - 1) /
                                          (2 * time_steps_per_job);
                                    if (job >=
                                        node_set.odd_time_data_nodes_per_job
                                            .size()) {
                                        node_set.odd_time_data_nodes_per_job
                                            .resize(job + 1);
                                    }
                                    node_set.odd_time_data_nodes_per_job[job]
                                        .push_back(node);
                                }
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
            for (int i = 0; i < timer_nodes.size(); i++) {
                const auto& timer = timer_nodes.at(i);
                this->sample_from_posterior({nullptr}, timer, false);

                const auto& reverse_timer =
                    timer_nodes.at(timer_nodes.size() - i - 1);
                dynamic_pointer_cast<TimerNode>(reverse_timer)
                    ->update_backward_assignment();
            }
        }

        void GibbsSampler::fill_initial_samples(
            const shared_ptr<gsl_rng>& random_generator) {
            // Observable nodes are already frozen by the Gibbs sampler thus
            // there's no need to add them as data in the ancestral sampler.
            AncestralSampler initial_sampler(this->model);
            initial_sampler.set_time_steps_per_sample(
                this->time_steps_per_sample);
            initial_sampler.set_num_in_plate_samples(
                this->num_in_plate_samples);
            initial_sampler.sample(random_generator, 1);
        }

        void GibbsSampler::init_samples_storage(
            const unordered_set<string>& sampled_node_labels) {

            // If there's no observation for a node in a specific time step,
            // this might be inferred by the value in the column that
            // represent such time step in the matrix of samples as it's
            // going to be filled with the original value = -1. Therefore,
            // all the matrices of samples have the same size, regardless of
            // the node's initial time step.
            for (const auto& node_label : sampled_node_labels) {
                if (!EXISTS(node_label, this->node_label_to_samples)) {
                    const auto& metadata =
                        this->model->get_metadata_of(node_label);
                    int sample_size = metadata->get_sample_size();
                    int time_steps = this->model->get_time_steps();
                    int rows = this->num_samples;
                    if (metadata->is_in_plate()) {
                        rows = this->num_in_plate_samples;
                    }
                    this->node_label_to_samples[node_label] =
                        Tensor3::constant(sample_size, rows, time_steps, -1);
                }
            }
        }

        void GibbsSampler::sample_parallel_time_nodes(
            const std::vector<std::shared_ptr<gsl_rng>>&
                random_generators_per_job,
            const std::vector<std::vector<std::shared_ptr<Node>>>&
                nodes_per_job) {

            if (nodes_per_job.empty()) {
                return;
            }

            if (nodes_per_job.size() == 1) {
                // Avoid the overhead of creating a new thread and use the main
                // one.
                this->run_sample_from_posterior_thread(
                    random_generators_per_job.at(0), nodes_per_job.at(0));
            }
            else {
                vector<thread> node_threads;
                for (int job = 0; job < nodes_per_job.size(); job++) {
                    thread node_thread(
                        &GibbsSampler::run_sample_from_posterior_thread,
                        this,
                        random_generators_per_job.at(job),
                        nodes_per_job.at(job));
                    node_threads.push_back(move(node_thread));
                }

                for (auto& node_thread : node_threads) {
                    node_thread.join();
                }
            }
        }

        void GibbsSampler::run_sample_from_posterior_thread(
            const std::shared_ptr<gsl_rng>& random_generator,
            const std::vector<std::shared_ptr<Node>>& nodes) {

            for (auto& node : nodes) {
                this->sample_from_posterior({random_generator}, node);
            }
        }

        void GibbsSampler::sample_from_posterior(
            const vector<shared_ptr<gsl_rng>>& random_generator_per_job,
            const shared_ptr<Node>& node,
            bool update_sufficient_statistics) {

            shared_ptr<RandomVariableNode> rv_node =
                dynamic_pointer_cast<RandomVariableNode>(node);

            if (!rv_node->is_frozen()) {
                Eigen::MatrixXd sample = rv_node->sample_from_posterior(
                    random_generator_per_job, this->time_steps_per_sample);

                if (rv_node->get_metadata()->is_timer()) {
                    dynamic_pointer_cast<TimerNode>(rv_node)
                        ->set_forward_assignment(sample);
                }
                else {
                    rv_node->set_assignment(sample);
                }

                this->keep_sample(rv_node, sample);
            }

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

        void
        GibbsSampler::keep_sample(const shared_ptr<RandomVariableNode>& node,
                                  const Eigen::MatrixXd& sample) {

            if (this->step_counter < this->burn_in_period) {
                return;
            }

            scoped_lock lock(*this->keep_sample_mutex);
            const string& node_label = node->get_metadata()->get_label();
            int time_step = node->get_time_step();

            if (this->step_counter >= this->burn_in_period) {
                //                if (sample.rows() == 1) {
                this->sampled_node_labels.insert(node_label);
                for (int i = 0; i < sample.cols(); i++) {
                    auto& samples_per_class =
                        this->node_label_to_samples.at(node_label)[i];
                    if (sample.rows() == 1) {
                        samples_per_class(this->step_counter -
                                              this->burn_in_period,
                                          time_step) = sample(0, i);
                    }
                    else {
                        // In-plate nodes always generate a matrix of samples
                        // with with as many rows as the number of data points.
                        samples_per_class.col(time_step) = sample.col(i);
                    }
                }
            }
        }

        void GibbsSampler::sample_sequential_time_nodes(
            const std::vector<std::shared_ptr<gsl_rng>>&
                random_generator_per_job,
            const std::vector<std::shared_ptr<Node>>& nodes) {

            for (auto& node : nodes) {
                // We only update the sufficient statistics for timer
                // nodes after, when all the time controlled nodes have
                // been sampled.
                bool upd_suff_stat = !node->get_metadata()->is_timer();
                this->sample_from_posterior(
                    random_generator_per_job, node, upd_suff_stat);
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
            }
        }

        void GibbsSampler::update_timer_backward_assignment(
            const std::vector<std::shared_ptr<Node>>& timer_nodes) {
            for (int j = timer_nodes.size() - 1; j >= 0; j--) {
                const auto& timer =
                    dynamic_pointer_cast<TimerNode>(timer_nodes.at(j));
                timer->update_backward_assignment();
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

        void GibbsSampler::get_info(nlohmann::json& json) const {
            json["name"] = "gibbs";
            json["burn_in"] = this->burn_in_period;
        }

        unique_ptr<Sampler> GibbsSampler::clone(bool unroll_model) const {
            unique_ptr<Sampler> new_sampler = make_unique<GibbsSampler>(
                this->model, this->burn_in_period, this->num_jobs);
            // Clone the model and the nodes in it
            new_sampler->set_model(
                make_shared<DynamicBayesNet>(this->model->clone(unroll_model)));

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

        void GibbsSampler::prepare() {
            Sampler::prepare();
            this->node_label_to_samples.clear();
        }

    } // namespace model
} // namespace tomcat
