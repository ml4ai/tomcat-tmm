#pragma once

#include "utils/Definitions.h"

#include "pipeline/estimation/EstimationProcess.h"

namespace tomcat {
    namespace model {

        /**
         * Class responsible for computing estimates for a model in an offline
         * fashion. The test data received from the pipeline is used in batch to
         * compute the estimates.
         */
        class OfflineEstimation : public EstimationProcess {
          public:
            //------------------------------------------------------------------
            // Constructors & Destructor
            //------------------------------------------------------------------

            /**
             * Creates an offline estimation process.
             */
            OfflineEstimation();

            /**
             * Creates an offline estimation process with an output stream to
             * write estimates to in an agent specific format at specific time
             * steps.
             */
            OfflineEstimation(std::ostream& output_stream);

            ~OfflineEstimation();

            //------------------------------------------------------------------
            // Copy & Move constructors/assignments
            //------------------------------------------------------------------
            OfflineEstimation(const OfflineEstimation& estimation);

            OfflineEstimation& operator=(const OfflineEstimation& estimation);

            OfflineEstimation(OfflineEstimation&&) = default;

            OfflineEstimation& operator=(OfflineEstimation&&) = default;

            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------
            void get_info(nlohmann::json& json) const override;

            void publish_last_estimates() override;

          private:
            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            /**
             * Copies data members from another offline estimation process.
             */
            void copy_estimation(const OfflineEstimation& estimation);


            std::shared_ptr<std::ostream> output_stream;
        };

    } // namespace model
} // namespace tomcat
