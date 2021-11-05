#pragma once

#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include <nlohmann/json.hpp>

#include "pipeline/estimation/Agent.h"
#include "utils/OnlineConfig.h"

namespace tomcat {
    namespace model {

        /**
         * Represents a generic class for reporting estimates in an specific
         * format, determined by some study.
         */
        class EstimateReporter {
          public:
            //------------------------------------------------------------------
            // Constructors & Destructor
            //------------------------------------------------------------------

            EstimateReporter();

            virtual ~EstimateReporter();

            //------------------------------------------------------------------
            // Copy & Move constructors/assignments
            //------------------------------------------------------------------

            // Copy constructor and assignment should be deleted to avoid
            // implicit slicing and loss of polymorphic behaviour in the
            // subclasses. To deep copy, the clone method must be used if any.
            EstimateReporter(const EstimateReporter&) = delete;

            EstimateReporter& operator=(const EstimateReporter&) = delete;

            EstimateReporter(EstimateReporter&&) = default;

            EstimateReporter& operator=(EstimateReporter&&) = default;

            //------------------------------------------------------------------
            // Virtual functions
            //------------------------------------------------------------------

            /**
             * Builds a heartbeat message to let the system know that the agent
             * is alive
             *
             * @param agent: agent
             * @return Heartbeat message
             */
            virtual nlohmann::json
            build_heartbeat_message(const AgentPtr& agent);

            /**
             * Builds any message that must be published when a mission
             * starts.
             *
             * @param agent: agent
             *
             * @return End of mission message
             */
            virtual nlohmann::json
            build_start_of_mission_message(const AgentPtr& agent);

            /**
             * Builds any message that must be published when a mission
             * finishes.
             *
             * @param agent: agent
             *
             * @return End of mission message
             */
            virtual nlohmann::json
            build_end_of_mission_message(const AgentPtr& agent);

            /**
             * Builds a message as a response to another message.
             *
             * @param agent: agent
             * @param request_message: message that requests a response
             * @param time_step: time step of the request
             *
             * @return Response message
             */
            virtual nlohmann::json
            build_message_by_request(const AgentPtr& agent,
                                     const nlohmann::json& request_message,
                                     int time_step);

            /**
             * Returns the right topic to publish a response to a request when
             * the report is used by an online agent.
             *
             * @return Response topic
             */
            virtual std::string get_request_response_topic(
                const nlohmann::json& request_message,
                const MessageBrokerConfiguration& broker_config);

            /**
             * Clear buffers and wait for a new mission.
             *
             */
            virtual void prepare();

            //------------------------------------------------------------------
            // Pure virtual functions
            //------------------------------------------------------------------

            /**
             * Builds a message to be published to a message bus containing
             * well formatted estimates computed for a given time step.
             *
             * @param agent: agent responsible for calculating the estimates
             * @param time_step: time step for which estimates were computed
             *
             * @return Messages.
             */
            virtual std::vector<nlohmann::json>
            translate_estimates_to_messages(const AgentPtr& agent,
                                            int time_step) = 0;

            /**
             * Builds a log message with a given text.
             *
             * @param agent: agent responsible for calculating the estimates
             *
             * @return Log message.
             */
            virtual nlohmann::json
            build_log_message(const AgentPtr& agent,
                              const std::string& log) const = 0;

          protected:
            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            /**
             * Gets the current timestamp as a string.
             *
             * @return Timestamp
             */
            std::string get_current_timestamp() const;

            /**
             * gets the timestamp after some seconds.
             *
             * @param initial_timestamp: timestamp before the elapsed time
             * @param elapsed_time: number of seconds to be added to the
             * initial timestamp
             *
             * @return Timestamp
             */
            std::string
            get_elapsed_timestamp(const std::string& initial_timestamp,
                                  int elapsed_time) const;
        };

    } // namespace model
} // namespace tomcat
