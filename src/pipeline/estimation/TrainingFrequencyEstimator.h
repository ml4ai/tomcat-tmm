#pragma once

#include <unordered_set>

#include "utils/Definitions.h"

#include "pipeline/estimation/PGMEstimator.h"

namespace tomcat {
    namespace model {

        /**
         * This estimator is based on the relative frequencies of the
         * observations over the training data for each time step on an unrolled
         * DBN. For instance, the probability of observing a value x in time
         * step t for a given node will be proportional to the number of times
         * the value x is observed in the training data for that node at time t.
         */
        class TrainingFrequencyEstimator : public PGMEstimator {
          public:

            inline static const std::string NAME = "training_frequency";

            //------------------------------------------------------------------
            // Constructors & Destructor
            //------------------------------------------------------------------

            /**
             * Creates an instance of a baseline estimator.
             *
             * @param model: DBN
             * @param inference_horizon: how many time steps in the future
             * estimations are going to be computed for
             * @param node_label: label of the node estimates are going to be
             * computed for
             * @param assignment: fixed assignment (for instance, estimates =
             * probability that the node assumes a value x, where x is the fixed
             * assignment). This parameter is optional when the inference
             * horizon is 0, but mandatory otherwise.
             */
            TrainingFrequencyEstimator(
                const std::shared_ptr<DynamicBayesNet>& model,
                int inference_horizon,
                const std::string& node_label,
                const Eigen::VectorXd& assignment = EMPTY_VECTOR);

            ~TrainingFrequencyEstimator();

            //------------------------------------------------------------------
            // Copy & Move constructors/assignments
            //------------------------------------------------------------------
            TrainingFrequencyEstimator(
                const TrainingFrequencyEstimator& estimator);

            TrainingFrequencyEstimator&
            operator=(const TrainingFrequencyEstimator& estimator);

            TrainingFrequencyEstimator(TrainingFrequencyEstimator&&) = default;

            TrainingFrequencyEstimator&
            operator=(TrainingFrequencyEstimator&&) = default;

            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------
            void estimate(const EvidenceSet& new_data) override;

            std::string get_name() const override;
        };

    } // namespace model
} // namespace tomcat
