#pragma once

#include "pipeline/estimation/custom_metrics/CustomSamplingMetric.h"

namespace tomcat {
    namespace model {

        /**
         * Represents a metric to estimate the mission final score given
         * samples generated until the end of the mission.
         */
        class FinalScore : public CustomSamplingMetric {
          public:
            const static int REGULAR_SCORE = 10;
            const static int CRITICAL_SCORE = 30;

            //------------------------------------------------------------------
            // Constructors & Destructor
            //------------------------------------------------------------------

            /**
             * Creates an instance of the final score metric.
             *
             */
            FinalScore();

            ~FinalScore();

            //------------------------------------------------------------------
            // Copy & Move constructors/assignments
            //------------------------------------------------------------------

            FinalScore(const FinalScore& final_score);

            FinalScore& operator=(const FinalScore& final_Score);

            FinalScore(FinalScore&&) = default;

            FinalScore& operator=(FinalScore&&) = default;

            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            /**
             * Use the samples generated for the Task node to count the
             * number of regular and critical victims saved and multiply that
             * by the proper score. We consider that a victim was saved,
             * every time the player stops saving a victim after subsequent
             * saving events. Therefore, we consider that all the savings
             * were carried out until the victim was saved. Unsuccessful
             * saving are not considered in this metric.
             *
             * @param sampler: sampler that holds the generated samples.
             * @param time_step: irrelevant for this metric. We always
             * recompute the final score using the full mission samples (from
             * time 0 until T).
             *
             * @return Single final score for the mission
             */
            std::vector<double>
            calculate(const std::shared_ptr<Sampler>& sampler,
                      int time_step) const override;

            //------------------------------------------------------------------
            // Getters & Setters
            //------------------------------------------------------------------

            //------------------------------------------------------------------
            // Data members
            //------------------------------------------------------------------
        };

    } // namespace model
} // namespace tomcat