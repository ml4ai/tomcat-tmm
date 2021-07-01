#pragma once

#include <memory>
#include <string>
#include <vector>

#include "pipeline/estimation/ASISTEstimateReporter.h"

namespace tomcat {
    namespace model {

        /**
         * Represents a TMM reporter for study 2 of the ASIST program.
         */
        class ASISTStudy2EstimateReporter : public ASISTEstimateReporter {
          public:
            //------------------------------------------------------------------
            // Constructors & Destructor
            //------------------------------------------------------------------

            ASISTStudy2EstimateReporter();

            virtual ~ASISTStudy2EstimateReporter();

            //------------------------------------------------------------------
            // Copy & Move constructors/assignments
            //------------------------------------------------------------------
            ASISTStudy2EstimateReporter(
                const ASISTStudy2EstimateReporter& agent);

            ASISTStudy2EstimateReporter&
            operator=(const ASISTStudy2EstimateReporter& agent);

            ASISTStudy2EstimateReporter(ASISTStudy2EstimateReporter&&) =
                default;

            ASISTStudy2EstimateReporter&
            operator=(ASISTStudy2EstimateReporter&&) = default;

          protected:
            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            nlohmann::json
            get_header_section(const AgentPtr& agent) const override;

            nlohmann::json get_msg_section(const AgentPtr& agent,
                                           int data_point) const override;

            nlohmann::json get_data_section(const AgentPtr& agent,
                                            int time_step,
                                            int data_point) const override;
        };

    } // namespace model
} // namespace tomcat
