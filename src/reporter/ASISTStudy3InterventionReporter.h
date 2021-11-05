#pragma once

#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "pipeline/estimation/Agent.h"
#include "reporter/EstimateReporter.h"

namespace tomcat {
    namespace model {

        /**
         * Represents a TMM reporter for study 3 of the ASIST program.
         */
        class ASISTStudy3InterventionReporter : public EstimateReporter {
          public:
            //------------------------------------------------------------------
            // Constructors & Destructor
            //------------------------------------------------------------------

            ASISTStudy3InterventionReporter();

            virtual ~ASISTStudy3InterventionReporter();

            //------------------------------------------------------------------
            // Copy & Move constructors/assignments
            //------------------------------------------------------------------
            ASISTStudy3InterventionReporter(
                const ASISTStudy3InterventionReporter& agent);

            ASISTStudy3InterventionReporter&
            operator=(const ASISTStudy3InterventionReporter& agent);

            ASISTStudy3InterventionReporter(ASISTStudy3InterventionReporter&&) =
                default;

            ASISTStudy3InterventionReporter&
            operator=(ASISTStudy3InterventionReporter&&) = default;

            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------
            nlohmann::json
            build_heartbeat_message(const AgentPtr& agent) override;

            nlohmann::json
            build_start_of_mission_message(const AgentPtr& agent) override;

            nlohmann::json
            build_end_of_mission_message(const AgentPtr& agent) override;

            nlohmann::json
            build_message_by_request(const AgentPtr& agent,
                                     const nlohmann::json& request_message,
                                     int time_step) override;

            std::string get_request_response_topic(
                const nlohmann::json& request_message,
                const MessageBrokerConfiguration& broker_config) override;

            std::vector<nlohmann::json>
            translate_estimates_to_messages(const AgentPtr& agent,
                                            int time_step) override;

            nlohmann::json
            build_log_message(const AgentPtr& agent,
                              const std::string& log) const override;

          protected:
            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            nlohmann::json
            get_header_section(const AgentPtr& agent,
                               const std::string& message_type) const;

            nlohmann::json get_msg_section(const AgentPtr& agent,
                                           const std::string& sub_type) const;

            nlohmann::json get_common_data_section(const AgentPtr& agent,
                                                   int time_step);

          private:
            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            /**
             * Returns ToMCAT introductory speech.
             *
             * @return: Speech
             */
            std::string get_introductory_speech() const;

            /**
             * Returns ToMCAT closing speech.
             *
             * @return: Speech
             */
            std::string get_closing_speech() const;

            /**
             * Get estimated team quality and confidence of the estimate in a
             * given time step.
             *
             * @param estimator: team quality estimator
             * @param time_step: time step
             *
             * @return: pair of team quality and confidence
             */
            std::pair<int, double>
            get_estimated_team_quality(const EstimatorPtr& estimator,
                                       int time_step) const;

            /**
             * Gets ToMCAT's speech to the players about the quality of their
             * team.
             *
             * @param quality: bad, medium or good
             * @param confidence: ToMCAT's confidence about the team's quality.
             *
             * @return: Speech
             */
            std::string get_tomcat_team_quality_speech(int quality,
                                                       int confidence) const;

            /**
             * Return a string with the participant ids separated by comma.
             *
             * @param agent: Agent
             *
             * @return: List of participant ids
             */
            nlohmann::json get_player_list(const AgentPtr& agent) const;

            /**
             * Calculates the timestamp at a given time step within the mission.
             *
             * @param agent: agent responsible for the predictions
             * @param time_step: time step
             *
             * @return Initial timestamp + elapsed time
             */
            std::string get_timestamp_at(const AgentPtr& agent,
                                         int time_step) const;

            /**
             * Calculates the milliseconds at a given time step within the
             * mission.
             *
             * @param agent: agent responsible for the predictions
             * @param time_step: time step
             *
             * @return Time step * step size * 1000
             */
            int get_milliseconds_at(const AgentPtr& agent, int time_step) const;

            /**
             * Builds a message to communicate information about a running
             * agent.
             *
             * @param agent: agent
             *
             * @return Version info message
             */
            nlohmann::json build_version_info_message(const AgentPtr& agent);

            /**
             * Builds a message to let the system know that the agent is up by
             * the system's request.
             */
            nlohmann::json
            build_rollcall_message(const AgentPtr& agent,
                                   const nlohmann::json& request_message,
                                   int time_step);

            //------------------------------------------------------------------
            // Data member
            //------------------------------------------------------------------
            int last_quality = -1;

            std::chrono::time_point<std::chrono::steady_clock>
                time_last_heartbeat;
        };

    } // namespace model
} // namespace tomcat
