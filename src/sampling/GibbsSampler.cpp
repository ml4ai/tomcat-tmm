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
            this->multitime_sampled_nodes = sampler.multitime_sampled_nodes;
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

            //            for (const auto& node : this->node_set.sampled_nodes)
            //            {
            //                cout << node->get_timed_name() << endl;
            //                cout << "-----------------" << endl;
            //                cout << node->get_assignment() << endl << endl;
            //            }
        }

        void GibbsSampler::sample_latent(
            const shared_ptr<gsl_rng>& random_generator) {

            this->init_sampling();
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

                // We sample multitime nodes here so that when this sampler
                // is being used as an estimator, these nodes can be updated
                // according to its new posterior given the sampled values
                // for the nodes in the new sampling range.
                this->sample_sequential_time_nodes(
                    random_generators_per_job, this->multitime_sampled_nodes);

                this->sample_parallel_time_nodes(
                    random_generators_per_job,
                    node_set.even_time_data_nodes_per_job);

                this->sample_parallel_time_nodes(
                    random_generators_per_job,
                    node_set.odd_time_data_nodes_per_job);

                this->sample_sequential_time_nodes(
                    random_generators_per_job,
                    node_set.nodes_sampled_in_sequence);

                if(this->step_counter >= this->burn_in_period) {
                    this->sample_sequential_time_nodes(
                        random_generators_per_job,
                        node_set.nodes_in_inference_horizon);
                }

                if (trainable) {
                    this->update_timer_sufficient_statistics(
                        node_set.timer_nodes);

                    this->sample_parallel_time_nodes(
                        random_generators_per_job,
                        node_set.parameter_nodes_per_job);
                }

                if (this->show_progress) {
                    ++(*progress);
                }
            }
        }

        void GibbsSampler::init_sampling() {
            this->step_counter = 0;
            this->max_time_step_to_sample =
                this->max_time_step_to_sample >= 0
                    ? this->max_time_step_to_sample
                    : this->model->get_time_steps() - 1;
        }

        GibbsSampler::NodeSet GibbsSampler::get_node_set() {
            NodeSet node_set;

            // We proceed from the root to the leaves so that child nodes
            // can update the sufficient statistics of parameter nodes
            // correctly given that parent nodes were already sampled.
            int params = 0;
            int from_time = this->min_time_step_to_sample;
            for (int t = from_time; t <= this->max_time_step_to_sample; t++) {
                for (auto& node :
                     this->model->get_nodes_in_topological_order_at(t)) {

                    const string node_label = node->get_metadata()->get_label();

                    if (node->get_metadata()->is_parameter()) {
                        // We don't sample parameter nodes that are frozen or if
                        // the sampler is not trainable
                        if (node->is_frozen() || !this->trainable) {
                            continue;
                        }

                        int job = params % this->num_jobs;
                        params++;
                        if (job >= node_set.parameter_nodes_per_job.size()) {
                            node_set.parameter_nodes_per_job.resize(job + 1);
                        }
                        node_set.parameter_nodes_per_job[job].push_back(node);
                        node_set.sampled_node_labels.insert(node_label);
                    }
                    else {
                        if (node->is_frozen() && !this->trainable) {
                            // If the sampler is trainable, we include frozen
                            // data nodes because they contribute to the
                            // sufficient statistics of the parameters'
                            // distributions.
                            continue;
                        }

                        node_set.sampled_node_labels.insert(node_label);

                        // Timer nodes and the nodes controlled by them cannot
                        // be processed in parallel as the size of the left and
                        // segments to which the controlled nodes belong to in
                        // random.
                        if (node->get_metadata()->is_timer()) {
                            node_set.timer_nodes.push_back(node);
                            node_set.nodes_sampled_in_sequence.push_back(node);
                        }
                        else {
                            if (node->has_timer()) {
                                node_set.nodes_sampled_in_sequence.push_back(
                                    node);
                            }
                            else {
                                if (node->get_metadata()->is_multitime()) {
                                    this->multitime_sampled_nodes.push_back(
                                        node);
                                }
                                else {
                                    int time_steps =
                                        this->max_time_step_to_sample -
                                        this->min_time_step_to_sample + 1;
                                    int time_steps_per_job =
                                        ceil(time_steps /
                                             (double)(2 * this->num_jobs));

                                    int job = 0;
                                    int time_step =
                                        node->get_time_step() -
                                        this->min_time_step_to_sample;
                                    if (time_step % 2 == 0) {
                                        job = time_step /
                                              (2 * time_steps_per_job);
                                        if (job >=
                                            node_set
                                                .even_time_data_nodes_per_job
                                                .size()) {
                                            node_set
                                                .even_time_data_nodes_per_job
                                                .resize(job + 1);
                                        }
                                        node_set
                                            .even_time_data_nodes_per_job[job]
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
                                        node_set
                                            .odd_time_data_nodes_per_job[job]
                                            .push_back(node);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Nodes in the inference horizon
            for (int t = this->max_time_step_to_sample + 1;
                 t <= this->max_time_step_to_sample + this->inference_horizon;
                 t++) {
                for (auto& node :
                     this->model->get_nodes_in_topological_order_at(t)) {
                    if (!node->get_metadata()->is_parameter()) {
                        const string& node_label =
                            node->get_metadata()->get_label();
                        node_set.sampled_node_labels.insert(node_label);
                        node_set.nodes_in_inference_horizon.push_back(node);

                        if (node->get_metadata()->is_timer()) {
                            node_set.timer_nodes.push_back(node);
                        }
                    }
                }
            }

            return node_set;
        }

        bool GibbsSampler::is_in_sampling_range(const RVNodePtr& node) const {
            int t = node->get_time_step();
            int t_min = this->min_time_step_to_sample;
            int t_max = this->max_time_step_to_sample;

            int t0 = node->get_metadata()->get_initial_time_step();
            bool multi_time = node->get_metadata()->is_multitime();

            return (t_min <= t && t <= t_max) || (multi_time && t0 < t_max);
        }

        bool
        GibbsSampler::is_in_inference_horizon(const RVNodePtr& node) const {
            int t = node->get_time_step();
            int t_min = this->max_time_step_to_sample + 1;
            int t_max = this->max_time_step_to_sample + this->inference_horizon;

            return (t_min <= t && t <= t_max);
        }

        void GibbsSampler::init_timers(
            const std::vector<std::shared_ptr<Node>> timer_nodes) {
            // The ancestral sampling will fill the duration of a segment
            // first, which means that the timer counter will be
            // filled backwards (d, d-1, d-2... instead of 0, 1, 2, ...).
            // Therefore we sample the timer nodes from their posterior to fix
            // the counters.
            for (auto& timer_node : timer_nodes) {
                this->sample_from_posterior({nullptr}, timer_node, false);
            }
        }

        void GibbsSampler::fill_initial_samples(
            const shared_ptr<gsl_rng>& random_generator) {
            // Observable nodes are already frozen by the Gibbs sampler thus
            // there's no need to add them as data in the ancestral sampler.
            AncestralSampler initial_sampler(this->model);
            initial_sampler.set_trainable(this->trainable);
            initial_sampler.set_min_time_step_to_sample(
                this->min_initialization_time_step);
            initial_sampler.set_max_time_step_to_sample(
                this->max_time_step_to_sample + this->inference_horizon);
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
                Eigen::MatrixXd sample(0, 0);
                if (node->get_metadata()->is_timer() || rv_node->has_timer() ||
                    !this->is_in_inference_horizon(rv_node) ||
                    this->inference_horizon == 0) {
                    // We estimate the posterior and sample from it. If this
                    // sampler is used for forward inference, we cache past
                    // posterior weights to speed up computation.
                    sample = rv_node->sample_from_posterior(
                        random_generator_per_job,
                        this->min_time_step_to_sample,
                        this->max_time_step_to_sample,
                        !this->trainable);
                }
                else {
                    // We don't estimate the posterior. We just sample from
                    // the current node's CPD. In estimation mode, the CPD's
                    // should have been already updated with posteriors from
                    // previous iterations of the estimator.
                    sample = rv_node->sample(random_generator_per_job,
                                             this->num_in_plate_samples);
                }

                if (rv_node->get_metadata()->is_timer()) {
                    dynamic_pointer_cast<TimerNode>(rv_node)
                        ->set_forward_assignment(sample);
                }
                else {
                    rv_node->set_assignment(sample);
                }

                this->keep_sample(rv_node, sample);
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

                const auto& reverse_timer = dynamic_pointer_cast<TimerNode>(
                    timer_nodes.at(timer_nodes.size() - j - 1));
                reverse_timer->update_backward_assignment(
                    this->max_time_step_to_sample);
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
            this->multitime_sampled_nodes.clear();
        }

    } // namespace model
} // namespace tomcat
