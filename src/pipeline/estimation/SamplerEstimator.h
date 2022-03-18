#pragma once

#include <memory>
#include <mutex>
#include <unordered_set>
#include <vector>

#include <nlohmann/json.hpp>

#include "pgm/inference/ParticleFilter.h"
#include "pipeline/estimation/PGMEstimator.h"
#include "sampling/AncestralSampler.h"
#include "sampling/Sampler.h"
#include "utils/Definitions.h"
#include "utils/Tensor3.h"

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
        class SamplerEstimator : public PGMEstimator {
          public:
            //------------------------------------------------------------------
            // Types, Enums & Constants
            //------------------------------------------------------------------

            enum FREQUENCY_TYPE { all, fixed, dynamic };

            //------------------------------------------------------------------
            // Constructors & Destructor
            //------------------------------------------------------------------

            SamplerEstimator() = default;

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
             * @param frequency_type: frequency at which estimates must be
             * computed
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
                             const Eigen::VectorXd& high = EMPTY_VECTOR,
                             FREQUENCY_TYPE frequency_type = all);

            ~SamplerEstimator() = default;

            //------------------------------------------------------------------
            // Copy & Move constructors/assignments
            //------------------------------------------------------------------
            SamplerEstimator(const SamplerEstimator& estimator);

            SamplerEstimator& operator=(const SamplerEstimator& estimator);

            SamplerEstimator(SamplerEstimator&&) = default;

            SamplerEstimator& operator=(SamplerEstimator&&) = default;

            //------------------------------------------------------------------
            // Static functions
            //------------------------------------------------------------------

            /**
             * Calculates a node's prior based on the priors of its parents.
             *
             * @param node: node
             *
             * @return Prior probability
             */
            static Eigen::VectorXd get_prior(const RVNodePtr& node);

            /**
             * factory function to create a new instance of a custom sampler
             * estimator from its name
             *
             * @param name: estimator name
             * @param model: model used for estimation
             * @param json_config: extra configuration
             * @param frequency_type: frequency at which estimates must be
             * computed

             *
             * @return Newly created instance of the custom estimator
             */
            static SamplerEstimatorPtr
            create_custom_estimator(const std::string& name,
                                    const DBNPtr& model,
                                    const nlohmann::json& json_config,
                                    FREQUENCY_TYPE frequency_type = all);

            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------
            void estimate(const EvidenceSet& new_data) override;

            std::string get_name() const override;

            bool is_binary_on_prediction() const override;

            /**
             * Checks whether the estimator does computation at a given time
             * step.
             *
             * @param data_point: data point
             * @param time_step: time step
             * @param new_data: new test data
             *
             * @return
             */
            bool does_estimation_at(int data_point,
                                    int time_step,
                                    const EvidenceSet& new_data) const;

            //------------------------------------------------------------------
            // Getters & Setters
            //------------------------------------------------------------------

            void set_fixed_steps(const std::unordered_set<int>& fixed_steps);

            //------------------------------------------------------------------
            // Virtual member functions
            //------------------------------------------------------------------

            /**
             * Compute and save probability estimates empirically from generated
             * particles.
             *
             * @param new_data: observations
             * @param particles: particles for the last time step processed
             * @param projected_particles: particles generated by transitioning
             * some time steps into the future
             * @param marginals: marginal probabilities over time for nodes that
             * were rao-blackwellized
             * @param data_point_idx: index of the data point for which
             * particles were generated
             * @param time_step: time step when the first particles in de set
             * were generated
             * @param filter: particle filter used to generate the samples
             */
            virtual void estimate(const EvidenceSet& new_data,
                                  const EvidenceSet& particles,
                                  const EvidenceSet& projected_particles,
                                  const EvidenceSet& marginals,
                                  int data_point_idx,
                                  int time_step,
                                  ParticleFilter& filter);

            /**
             * Checks whether a custom event is triggered at a given time
             * step and data point.
             *
             * @param data_point: data point
             * @param time_step: time step
             * @param new_data: new test data
             *
             * @return
             */
            virtual bool
            is_event_triggered_at(int data_point,
                                  int time_step,
                                  const EvidenceSet& new_data) const;

            /**
             * Make any necessary preparations before estimates for a new data
             * point start to be computed.
             *
             */
            virtual void prepare_for_the_next_data_point() const;

          protected:
            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            /**
             * Copies data member from another estimator.
             *
             * @param estimator: other estimator
             */
            void copy(const SamplerEstimator& estimator);

            /**
             * Append a new probability to the list of estimates.
             *
             * @param estimates_idx: index of the estimates matrix to use
             * @param data_point_idx: row of the estimates matrix being updated
             * @param time_step: column of the estimates matrix being updated
             * @param probability: probability estimate
             */
            void update_estimates(int estimates_idx,
                                  int data_point_idx,
                                  int time_step,
                                  double probability);

            /**
             * Append a new custom calculation to the list of custom data.
             *
             * @param estimates_idx: index of the estimates matrix to use
             * @param data_point_idx: row of the estimates matrix being updated
             * @param time_step: column of the estimates matrix being updated
             * @param probability: probability estimate
             */
            void update_custom_data(int estimates_idx,
                                    int data_point_idx,
                                    int time_step,
                                    double probability);

            /**
             * From a vector of sampled values from a categorical distribution.
             * Calculate the empirical probability of each category.
             *
             * @param samples: samples
             * @param cardinality: number of categories
             *
             * @return Empirical probability per category
             */
            Eigen::VectorXd
            calculate_probabilities_from_samples(const Eigen::VectorXd& samples,
                                                 int cardinality) const;

            //------------------------------------------------------------------
            // Data members
            //------------------------------------------------------------------

            FREQUENCY_TYPE frequency_type;
            std::unordered_set<int> fixed_steps;

          private:
            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            double get_probability_in_range(const Tensor3& samples,
                                            double low,
                                            double high) const;
        };

    } // namespace model
} // namespace tomcat
