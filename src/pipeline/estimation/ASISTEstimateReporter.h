#pragma once

#include <memory>
#include <string>
#include <vector>

#include "pipeline/estimation/EstimateReporter.h"

namespace tomcat {
    namespace model {

        /**
         * Represents a TMM reporter for the ASIST program.
         */
        class ASISTEstimateReporter : public EstimateReporter {
          public:
            //------------------------------------------------------------------
            // Constructors & Destructor
            //------------------------------------------------------------------

            ASISTEstimateReporter();

            virtual ~ASISTEstimateReporter();

            //------------------------------------------------------------------
            // Copy & Move constructors/assignments
            //------------------------------------------------------------------
            ASISTEstimateReporter(const ASISTEstimateReporter& reporter) =
                delete;

            ASISTEstimateReporter&
            operator=(const ASISTEstimateReporter& reporter) = delete;

            ASISTEstimateReporter(ASISTEstimateReporter&&) = default;

            ASISTEstimateReporter& operator=(ASISTEstimateReporter&&) = default;

            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------
            std::vector<nlohmann::json>
            translate_estimates_to_messages(const AgentPtr& agent,
                                 int time_step) const override;

            nlohmann::json
            build_log_message(const AgentPtr& agent,
                              const std::string& log) const override;

          protected:
            //------------------------------------------------------------------
            // Pure virtual functions
            //------------------------------------------------------------------

            /**
             * Fills the header section of a message to be publish
             *
             * @param agent: agent responsible for calculating the estimates
             *
             * @return Header section
             */
            virtual nlohmann::json
            get_header_section(const AgentPtr& agent) const = 0;

            /**
             * Returns a pair (state, prediction) msg section section of a message to be publish.
             *
             * @param agent: agent responsible for calculating the estimates
             * @param data_point: index of the data point if estimates were
             * computed for multiple set of observations
             *
             * @return Msg section
             */
            virtual std::pair<nlohmann::json, nlohmann::json>
            get_msg_section(const AgentPtr& agent, int data_point) const = 0;

            /**
             * Returns a pair (state, prediction) data section of a message to be publish.
             *
             * @param agent: agent responsible for calculating the estimates
             * @param time_step: time step to look for estimates
             * @param data_point: index of the data point if estimates were
             * computed for multiple set of observations
             *
             * @return Data section
             */
            virtual std::pair<nlohmann::json, nlohmann::json> get_data_section(
                const AgentPtr& agent, int time_step, int data_point) const = 0;
        };

    } // namespace model
} // namespace tomcat
