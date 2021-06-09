#pragma once

#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "utils/Definitions.h"

#include "pgm/inference/ParticleFilter.h"
#include "pipeline/estimation/SamplerEstimator.h"

namespace tomcat {
    namespace model {

        /**
         * This class represents an estimator that computes the
         * estimates by particle filter. Particles are generated at every call
         * to the estimate function. Sampler estimators (used for evaluation
         * purposes) compute estimates for a given node empirically from the
         * particles generated at every time step. Therefore, this estimator
         * cannot be used for evaluation but only for estimation purposes to
         * share computation among different nodes' estimates. The individual
         * sampler estimators can be used for evaluation purposes.
         */
        class ParticleFilterEstimator : public Estimator {
          public:
            //------------------------------------------------------------------
            // Constructors & Destructor
            //------------------------------------------------------------------

            /**
             * Creates an instance of a particle filter estimator.
             *
             * @param model: pre-trained dynamic bayes net
             * @param random_generator: random number generator
             * @param num_particles: number of particles to generate to
             * approximate the estimated distributions
             * @param num_jobs: number of threads used
             * @param variable_horizon_max_time_step: maximum time step to
             * project particles in case any of the base estimators requires a
             * variable horizon
             */
            ParticleFilterEstimator(
                const DBNPtr& model,
                int num_particles,
                const std::shared_ptr<gsl_rng>& random_generator,
                int num_jobs,
                int variable_horizon_max_time_step = 0);

            ~ParticleFilterEstimator();

            //------------------------------------------------------------------
            // Copy & Move constructors/assignments
            //------------------------------------------------------------------
            ParticleFilterEstimator(const ParticleFilterEstimator& estimator);

            ParticleFilterEstimator&
            operator=(const ParticleFilterEstimator& estimator);

            ParticleFilterEstimator(ParticleFilterEstimator&&) = default;

            ParticleFilterEstimator&
            operator=(ParticleFilterEstimator&&) = default;

            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------
            void prepare() override;

            void keep_estimates() override;

            void estimate(const EvidenceSet& new_data) override;

            void get_info(nlohmann::json& json) const override;

            std::string get_name() const override;

            bool is_computing_estimates_for(
                const std::string& node_label) const override;

            std::vector<std::shared_ptr<const Estimator>>
            get_base_estimators() const override;

            /**
             * Adds a sampler estimator to this compound estimator.
             *
             * @param estimator: sampler estimator
             */
            void add_base_estimator(
                const std::shared_ptr<SamplerEstimator>& estimator);

          private:
            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            /**
             * Copies data members from another estimator.
             *
             * @param estimator: other estimator
             */
            void copy(const ParticleFilterEstimator& estimator);

            //------------------------------------------------------------------
            // Data members
            //------------------------------------------------------------------

            ParticleFilter template_filter;

            std::vector<ParticleFilter> filter_per_data_point;

            int last_time_step = -1;

            // Max inference horizon among all base samplers
            int max_inference_horizon = 0;

            std::vector<std::shared_ptr<SamplerEstimator>> base_estimators;

            bool variable_horizon = false;

            int variable_horizon_max_time_step = 0;
        };

    } // namespace model
} // namespace tomcat