#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include "utils/Definitions.h"

#include "pipeline/estimation/Estimator.h"
#include "pipeline/estimation/SamplerEstimator.h"
#include "sampling/Sampler.h"

namespace tomcat {
    namespace model {

        /**
         * This class represents a compound estimator that computes the
         * estimates by using samples from the nodes' posterior distribution.
         * This estimator has a sampler that generates samples at every call to
         * the estimate function. Concrete estimators (used for evaluation
         * purposes) compute estimates for a given node by using the samples
         * generated by the sampler. Therefore, this estimator cannot be used
         * for evaluation but only for estimation purposes to share computation
         * among different nodes' estimates.
         */
        class CompoundSamplerEstimator : public Estimator {
          public:
            //------------------------------------------------------------------
            // Constructors & Destructor
            //------------------------------------------------------------------

            /**
             * Creates an instance of a sampler estimator.
             *
             * @param model: DBN
             * @param sampler: sampler responsible to generate samples from
             * the model
             * @param random_generator: random number generator
             * @param num_samples: number of samples to generate to
             * approximate the estimated distributions
             */
            CompoundSamplerEstimator(
                const std::shared_ptr<DynamicBayesNet>& model,
                const std::shared_ptr<Sampler>& sampler,
                const std::shared_ptr<gsl_rng>& random_generator,
                int num_samples);

            ~CompoundSamplerEstimator();

            //------------------------------------------------------------------
            // Copy & Move constructors/assignments
            //------------------------------------------------------------------
            CompoundSamplerEstimator(const CompoundSamplerEstimator& estimator);

            CompoundSamplerEstimator&
            operator=(const CompoundSamplerEstimator& estimator);

            CompoundSamplerEstimator(CompoundSamplerEstimator&&) = default;

            CompoundSamplerEstimator&
            operator=(CompoundSamplerEstimator&&) = default;

            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------
            void prepare() override;

            void estimate(const EvidenceSet& new_data) override;

            void get_info(nlohmann::json& json) const override;

            std::string get_name() const override;

            /**
             * Clear estimates and unfreeze observable nodes.
             */
            void cleanup() override;

            /**
             * Adds a concrete estimator to this compound estimator.
             *
             * @param estimator: concrete estimator
             */
            void add_estimator(const SamplerEstimator& estimator);

          private:
            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            /**
             * Copies data member from another estimator.
             *
             * @param estimator: other estimator
             */
            void copy(const CompoundSamplerEstimator& estimator);

            /**
             * Unfreeze nodes that were frozen they had data.
             */
            void unfreeze_observable_nodes();

            /**
             * Adds a new data collection to the nodes from a model.
             *
             * @param new_data: new data
             * @param time_step: data's column (time step) to use
             */
            void add_data_to_nodes(const EvidenceSet& new_data, int time_step);

            //------------------------------------------------------------------
            // Data members
            //------------------------------------------------------------------

            std::shared_ptr<gsl_rng> random_generator;

            int num_samples;

            // Next time step to generate samples to.
            int next_time_step = 0;

            std::shared_ptr<Sampler> sampler;

            std::vector<SamplerEstimator> estimators;

            std::unordered_map<std::string, RVNodePtr> nodes_to_unfreeze;
        };

    } // namespace model
} // namespace tomcat
