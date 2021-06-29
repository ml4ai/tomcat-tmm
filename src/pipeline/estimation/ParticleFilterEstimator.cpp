#include "ParticleFilterEstimator.h"

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
        ParticleFilterEstimator::ParticleFilterEstimator(
            const DBNPtr& model,
            int num_particles,
            const std::shared_ptr<gsl_rng>& random_generator,
            int num_jobs,
            int variable_horizon_max_time_step)
            : Estimator(model), num_particles(num_particles),
              random_generator(random_generator), num_jobs(num_jobs),
              variable_horizon_max_time_step(variable_horizon_max_time_step) {}

        ParticleFilterEstimator::~ParticleFilterEstimator() {}

        //----------------------------------------------------------------------
        // Copy & Move constructors/assignments
        //----------------------------------------------------------------------
        ParticleFilterEstimator::ParticleFilterEstimator(
            const ParticleFilterEstimator& estimator) {
            Estimator::copy_estimator(estimator);
            this->copy(estimator);
        }

        ParticleFilterEstimator& ParticleFilterEstimator::operator=(
            const ParticleFilterEstimator& estimator) {
            Estimator::copy_estimator(estimator);
            this->copy_estimator(estimator);
            return *this;
        }

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        void ParticleFilterEstimator::copy(
            const ParticleFilterEstimator& estimator) {
            this->num_particles = estimator.num_particles;
            this->random_generator = estimator.random_generator;
            this->num_jobs = estimator.num_jobs;
            this->max_inference_horizon = estimator.inference_horizon;
            this->last_time_step = estimator.last_time_step;
            this->base_estimators = estimator.base_estimators;
            this->variable_horizon = estimator.variable_horizon;
            this->variable_horizon_max_time_step =
                estimator.variable_horizon_max_time_step;
        }

        void ParticleFilterEstimator::prepare() {
            for (auto& base_estimator : this->base_estimators) {
                base_estimator->prepare();
            }
            this->last_time_step = -1;
        }

        void ParticleFilterEstimator::keep_estimates() {
            Estimator::keep_estimates();

            for (auto& base_estimator : this->base_estimators) {
                base_estimator->keep_estimates();
            }
        }

        void ParticleFilterEstimator::estimate(const EvidenceSet& new_data) {
            unique_ptr<boost::progress_display> progress;

            bool show_filter_progress = false;
            if (this->show_progress) {
                cout << "\nEmpirically computing estimations...\n";
                if (new_data.get_num_data_points() == 1) {
                    show_filter_progress = true;
                    this->show_progress = false;
                }
                else {
                    progress = make_unique<boost::progress_display>(
                        new_data.get_num_data_points());
                }
            }

            for (int d = 0; d < new_data.get_num_data_points(); d++) {
                ParticleFilter filter(*this->model,
                                      this->num_particles,
                                      this->random_generator,
                                      this->num_jobs);
                filter.set_show_progress(show_filter_progress);

                EvidenceSet single_point_data =
                    new_data.get_single_point_data(d);

                if (this->max_inference_horizon == 0 &&
                    !this->variable_horizon) {
                    // Generate particles for all time steps and compute
                    // estimates in the end
                    auto [particles, marginals] =
                        filter.generate_particles(single_point_data);
                    EvidenceSet projected_particles;

                    for (const auto& base_estimator : this->base_estimators) {
                        base_estimator->estimate(particles,
                                                 projected_particles,
                                                 marginals,
                                                 d,
                                                 this->last_time_step + 1);
                    }
                }
                else {
                    for (int t = 0; t < single_point_data.get_time_steps();
                         t++) {
                        // Generate particles and projections for each time
                        // step.

                        EvidenceSet single_time_data =
                            single_point_data.get_single_time_data(t);

                        auto [particles, marginals] =
                            filter.generate_particles(single_time_data);

                        int time_steps_ahead =
                            this->variable_horizon
                                ? this->variable_horizon_max_time_step -
                                      (this->last_time_step + t + 1)
                                : this->max_inference_horizon;

                        EvidenceSet projected_particles =
                            filter.forward_particles(time_steps_ahead);

                        for (const auto& base_estimator :
                             this->base_estimators) {
                            base_estimator->estimate(particles,
                                                     projected_particles,
                                                     marginals,
                                                     d,
                                                     this->last_time_step + 1 +
                                                         t);
                        }
                    }
                }

                if (this->show_progress) {
                    ++(*progress);
                }
            }

            this->last_time_step += new_data.get_time_steps();
        }

        void ParticleFilterEstimator::get_info(nlohmann::json& json) const {
            json["name"] = this->get_name();
        }

        string ParticleFilterEstimator::get_name() const {
            return "particle_filter";
        }

        bool ParticleFilterEstimator::is_computing_estimates_for(
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
        ParticleFilterEstimator::get_base_estimators() const {
            vector<shared_ptr<const Estimator>> base_estimators;
            for (const auto& base_estimator : this->base_estimators) {
                base_estimators.push_back(base_estimator);
            }
            return base_estimators;
        }

        void ParticleFilterEstimator::add_base_estimator(
            const shared_ptr<SamplerEstimator>& estimator) {

            this->base_estimators.push_back(estimator);

            this->max_inference_horizon =
                max(this->max_inference_horizon,
                    estimator->get_inference_horizon());

            if (estimator->get_inference_horizon() < 0) {
                this->variable_horizon = true;
            }
        }

    } // namespace model
} // namespace tomcat
