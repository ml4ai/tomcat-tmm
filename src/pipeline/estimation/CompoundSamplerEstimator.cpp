#include "CompoundSamplerEstimator.h"

#include <iostream>
#include <thread>

#include "utils/Multithreading.h"

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        CompoundSamplerEstimator::CompoundSamplerEstimator(
            const shared_ptr<DynamicBayesNet>& model,
            const shared_ptr<Sampler>& sampler,
            const std::shared_ptr<gsl_rng>& random_generator,
            int num_samples,
            int num_jobs)
            : Estimator(model), num_samples(num_samples) {

            if (num_jobs < 1) {
                throw invalid_argument("The number of jobs has to be at least "
                                       "one.");
            }

            // Computations are performed in cloned samplers and models.
            for (int i = 0; i < num_jobs; i++) {
                shared_ptr<Sampler> new_sampler = move(sampler->clone());
                new_sampler->set_num_in_plate_samples(1);
                this->sampler_per_job.push_back(new_sampler);
            }

            this->random_generator_per_job =
                split_random_generator(random_generator, num_jobs);
        }

        CompoundSamplerEstimator::~CompoundSamplerEstimator() {}

        //----------------------------------------------------------------------
        // Copy & Move constructors/assignments
        //----------------------------------------------------------------------
        CompoundSamplerEstimator::CompoundSamplerEstimator(
            const CompoundSamplerEstimator& estimator) {
            Estimator::copy_estimator(estimator);
            this->copy_estimator(estimator);
        }

        CompoundSamplerEstimator& CompoundSamplerEstimator::operator=(
            const CompoundSamplerEstimator& estimator) {
            Estimator::copy_estimator(estimator);
            this->copy_estimator(estimator);
            return *this;
        }

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        void CompoundSamplerEstimator::copy(
            const CompoundSamplerEstimator& estimator) {
            this->next_time_step = estimator.next_time_step;
            this->random_generator_per_job = estimator.random_generator_per_job;
            this->sampler_per_job = estimator.sampler_per_job;
        }

        void CompoundSamplerEstimator::prepare() {
            Estimator::prepare();
            Estimator::cleanup();
            this->next_time_step = 0;
            for (const auto& sampler : this->sampler_per_job) {
                sampler->get_model()->unroll(this->inference_horizon, true);
            }
        }

        void CompoundSamplerEstimator::estimate(const EvidenceSet& new_data) {
            int num_jobs = this->random_generator_per_job.size();
            auto processing_blocks = get_parallel_processing_blocks(
                num_jobs, new_data.get_num_data_points());

            int time_steps = this->next_time_step + new_data.get_time_steps();
            for (int t = this->next_time_step; t < time_steps; t++) {
                vector<thread> threads;
                for (int i = 0; i < num_jobs; i++) {
                    const auto& sampler = this->sampler_per_job.at(i);
                    sampler->get_model()->expand(1);
                    if (t > this->next_time_step) {
                        // Nodes smaller than this->inference_horizon + t were
                        // already sampled in the previous iteration.
                        sampler->set_min_initialization_time_step(
                            this->inference_horizon + t);
                    }
                    sampler->prepare();

                    if (num_jobs == 1) {
                        // Run in the main thread
                        run_estimation_thread(
                            this->random_generator_per_job.at(i),
                            sampler,
                            new_data,
                            t,
                            processing_blocks.at(i).first,
                            processing_blocks.at(i).second);
                    }
                    else {
                        thread estimation_thread(
                            &CompoundSamplerEstimator::run_estimation_thread,
                            this,
                            this->random_generator_per_job.at(i),
                            sampler,
                            new_data,
                            t,
                            processing_blocks.at(i).first,
                            processing_blocks.at(i).second);
                        threads.push_back(move(estimation_thread));
                    }

                    for (auto& estimation_thread : threads) {
                        estimation_thread.join();
                    }
                }
            }

            this->next_time_step = time_steps;
        }

        void CompoundSamplerEstimator::run_estimation_thread(
            const std::shared_ptr<gsl_rng>& random_generator,
            const std::shared_ptr<Sampler>& sampler,
            const EvidenceSet& new_data,
            int time_step,
            int initial_data_idx,
            int final_data_idx) {

            int data_size = final_data_idx - initial_data_idx;

            vector<vector<Eigen::VectorXd>> probs_per_estimator_and_class(
                this->estimators.size());
            for (int d = initial_data_idx; d < final_data_idx; d++) {
                // Observations for a single data point at the time step
                // for which we are computing inferences
                EvidenceSet data =
                    new_data.at(d, time_step - this->next_time_step);
                this->add_data_to_nodes(sampler->get_model(), data, time_step);

                sampler->sample(random_generator, this->num_samples);

                for (int i = 0; i < this->estimators.size(); i++) {
                    // Use the samples generated by the sampler to compute
                    // concrete estimates
                    vector<double> probs =
                        this->estimators.at(i)->estimate(sampler, d, time_step);

                    if (probs_per_estimator_and_class.at(i).empty()) {
                        probs_per_estimator_and_class.at(i) =
                            vector<Eigen::VectorXd>(probs.size());
                    }

                    for (int k = 0; k < probs.size(); k++) {
                        if (probs_per_estimator_and_class.at(i).at(k).size() ==
                            0) {
                            probs_per_estimator_and_class.at(i).at(k) =
                                Eigen::VectorXd(data_size);
                        }

                        probs_per_estimator_and_class.at(i).at(k)(
                            d - initial_data_idx) = probs.at(k);
                    }
                }
            }

            // Dump the estimates vector to the matrix of estimates for each
            // estimator.
            for (int i = 0; i < this->estimators.size(); i++) {
                this->estimators.at(i)->set_estimates(
                    probs_per_estimator_and_class.at(i),
                    initial_data_idx,
                    final_data_idx,
                    time_step);
            }
        }

        void CompoundSamplerEstimator::add_data_to_nodes(
            const shared_ptr<DynamicBayesNet>& model,
            const EvidenceSet& new_data,
            int time_step) {

            for (const auto& label : new_data.get_node_labels()) {
                Eigen::MatrixXd data = new_data[label](0, 0);
                const RVNodePtr node = dynamic_pointer_cast<RandomVariableNode>(
                    model->get_node(label, time_step));
                if (node) {
                    node->unfreeze();
                    node->set_assignment(data);
                    node->freeze();
                }
            }
        }

        void CompoundSamplerEstimator::get_info(nlohmann::json& json) const {
            json["name"] = this->get_name();
            this->sampler_per_job[0]->get_info(json["sampler"]);
        }

        string CompoundSamplerEstimator::get_name() const { return "sampler"; }

        void CompoundSamplerEstimator::add_estimator(
            const shared_ptr<SamplerEstimator>& estimator) {

            this->estimators.push_back(estimator);

            this->inference_horizon = max(this->inference_horizon,
                                          estimator->get_inference_horizon());
        }

    } // namespace model
} // namespace tomcat
