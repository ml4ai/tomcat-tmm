#include "CompoundSamplerEstimator.h"

#include <iostream>
#include <thread>

#include <boost/progress.hpp>

#include "utils/Multithreading.h"

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        CompoundSamplerEstimator::CompoundSamplerEstimator(
            const shared_ptr<Sampler>& sampler,
            const std::shared_ptr<gsl_rng>& random_generator,
            int num_samples)
            : Estimator(sampler->get_model()), sampler(sampler),
              random_generator(random_generator), num_samples(num_samples) {

            this->sampler->set_trainable(false);
            this->sampler->set_num_in_plate_samples(this->num_samples);
            this->sampler->set_show_progress(false);
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
            this->random_generator = estimator.random_generator;
            this->sampler = estimator.sampler;
        }

        void CompoundSamplerEstimator::cleanup() {
            Estimator::cleanup();

            for(auto& base_estimator : this->base_estimators) {
                base_estimator->cleanup();
            }
        }

        void CompoundSamplerEstimator::prepare() {
            Estimator::prepare();

            for(auto& base_estimator : this->base_estimators) {
                base_estimator->prepare();
            }

            this->next_time_step = 0;
            this->model->expand(this->inference_horizon);
            this->sampler->set_inference_horizon(this->inference_horizon);
            this->sampler->prepare();
        }

        void CompoundSamplerEstimator::keep_estimates() {
            Estimator::keep_estimates();

            for(auto& base_estimator : this->base_estimators) {
                base_estimator->keep_estimates();
            }
        }

        void CompoundSamplerEstimator::estimate(const EvidenceSet& new_data) {
            unique_ptr<boost::progress_display> progress;
            if (this->show_progress) {
                cout << "\nEmpirically computing estimations...\n";
                progress = make_unique<boost::progress_display>(
                    new_data.get_num_data_points()*new_data.get_time_steps());
            }

            int max_time_step =
                this->next_time_step + new_data.get_time_steps();
            for (int d = 0; d < new_data.get_num_data_points(); d++) {
                for (int t = this->next_time_step; t < max_time_step; t++) {
                    this->sampler->set_min_initialization_time_step(t);
                    this->sampler->set_min_time_step_to_sample(t);
                    this->sampler->set_max_time_step_to_sample(t);

                    // Observations for a single data point at the time step
                    // for which we are computing inferences
                    EvidenceSet data = new_data.at(d, t - this->next_time_step);
                    this->add_data_to_nodes(data, t);

                    this->sampler->sample(this->random_generator, 1);

                    for (int i = 0; i < this->base_estimators.size(); i++) {
                        this->base_estimators.at(i)->estimate(this->sampler, d, t);
                    }

                    if (this->show_progress) {
                        ++(*progress);
                    }
                }
            }

            this->next_time_step = max_time_step;
        }

        void
        CompoundSamplerEstimator::add_data_to_nodes(const EvidenceSet& new_data,
                                                    int time_step) {

            for (const auto& label : new_data.get_node_labels()) {
                const RVNodePtr node = dynamic_pointer_cast<RandomVariableNode>(
                    this->model->get_node(label, time_step));
                if (node) {
                    Eigen::MatrixXd data = new_data[label](0, 0);
                    data = data.replicate(this->num_samples, 1);
                    node->unfreeze();
                    node->set_assignment(data);
                    node->freeze();
                }
            }
        }

        void CompoundSamplerEstimator::get_info(nlohmann::json& json) const {
            json["name"] = this->get_name();
            this->sampler->get_info(json["sampler"]);
        }

        string CompoundSamplerEstimator::get_name() const { return "sampler"; }

        bool CompoundSamplerEstimator::is_computing_estimates_for(
            const std::string& node_label) const {
            bool is_estimating = false;

            for (const auto& estimator : this->base_estimators) {
                is_estimating =
                    estimator->is_computing_estimates_for(node_label);
                if (is_estimating)
                    break;
            }

            return is_estimating;
        }

        vector<shared_ptr<const Estimator>>
        CompoundSamplerEstimator::get_base_estimators() const {
            vector<shared_ptr<const Estimator>> base_estimators;
            for(const auto& base_estimator : this->base_estimators) {
                base_estimators.push_back(base_estimator);
            }
            return base_estimators;
        }

        void CompoundSamplerEstimator::add_base_estimator(
            const shared_ptr<SamplerEstimator>& estimator) {

            this->base_estimators.push_back(estimator);

            this->inference_horizon = max(this->inference_horizon,
                                          estimator->get_inference_horizon());
        }

    } // namespace model
} // namespace tomcat
