#pragma once

#include <memory>
#include <vector>

#include "utils/Definitions.h"

#include "pipeline/estimation/Estimator.h"
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
        class SamplerEstimator : public Estimator {
          public:
            //------------------------------------------------------------------
            // Constructors & Destructor
            //------------------------------------------------------------------

            /**
             * Creates an instance of a sampler estimator.
             *
             * @param model: DBN
             * @param inference_horizon: how many time steps in the future
             * estimations are going to be computed for
             * @param node_label: label of the node estimates are going to be
             * computed for
             * @param low: fixed assignment (for instance, estimates =
             * probability that the node assumes a value >= x, where x is the
             * fixed assignment). If the node's distribution is discrete,
             * will be used to return p(x = low) in the inference horizon
             * @param high: fixed assignment (for instance, estimates =
             * probability that the node assumes a value <= x, where x is the
             * fixed assignment).
             *
             * Note: If neither low or high are informed, if the node's
             * distribution is discrete it will return the full distribution
             * (probabilities for each assignment of the node). Otherwise, it
             * will return 1.
             */
            SamplerEstimator(const std::shared_ptr<DynamicBayesNet>& model,
                             int inference_horizon,
                             const std::string& node_label,
                             const Eigen::VectorXd& low = EMPTY_VECTOR,
                             const Eigen::VectorXd& high = EMPTY_VECTOR);

            ~SamplerEstimator();

            //------------------------------------------------------------------
            // Copy & Move constructors/assignments
            //------------------------------------------------------------------
            SamplerEstimator(const SamplerEstimator& estimator);

            SamplerEstimator& operator=(const SamplerEstimator& estimator);

            SamplerEstimator(SamplerEstimator&&) = default;

            SamplerEstimator& operator=(SamplerEstimator&&) = default;

            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------
            void prepare() override;

            void estimate(const EvidenceSet& new_data) override;

            void get_info(nlohmann::json& json) const override;

            std::string get_name() const override;

            //------------------------------------------------------------------
            // Getters & Setters
            //------------------------------------------------------------------
            void set_sampler(const std::shared_ptr<Sampler>& sampler);

          private:
            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            /**
             * Copies data member from another estimator.
             *
             * @param estimator: other estimator
             */
            void copy(const SamplerEstimator& estimator);

            //------------------------------------------------------------------
            // Data members
            //------------------------------------------------------------------

            // Next time step to generate samples to.
            int next_time_step = 0;

            std::shared_ptr<Sampler> sampler;
        };

    } // namespace model
} // namespace tomcat