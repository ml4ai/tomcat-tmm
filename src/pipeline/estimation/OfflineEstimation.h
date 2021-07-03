#pragma once

#include <iostream>

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

            /*
             * Creates an offline estimator with no defined estimate report.
             * Estimates are still written as raw probabilities in the get_info
             * function.
             *
             * @param agent: agent used in the estimation
             *
             */
            OfflineEstimation(const AgentPtr& agent);

            /**
             * Creates an offline estimation process with an output stream to
             * write estimates to in an agent specific format at specific time
             * steps.
             *
             * @param agent: agent used in the estimation
             * @param reporter: class responsible for reporting estimates
             * computed during the process
             * @param report_filepath: path of the written report file. If the
             * path is blank, the report will be printed to the terminal.
             */
            OfflineEstimation(const AgentPtr& agent,
                              const EstimateReporterPtr& reporter,
                              const std::string& report_filepath);

            ~OfflineEstimation();

            //------------------------------------------------------------------
            // Copy & Move constructors/assignments
            //------------------------------------------------------------------

            // It cannot be copied because ofstreams cannot be copied.
            OfflineEstimation(const OfflineEstimation& estimation) = delete;

            OfflineEstimation&
            operator=(const OfflineEstimation& estimation) = delete;

            OfflineEstimation(OfflineEstimation&&) = delete;

            OfflineEstimation& operator=(OfflineEstimation&&) = delete;

            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------
            void get_info(nlohmann::json& json) const override;

            void publish_last_estimates() override;

          private:
            //------------------------------------------------------------------
            // Data member
            //------------------------------------------------------------------

            std::ofstream report_file;
        };

    } // namespace model
} // namespace tomcat
