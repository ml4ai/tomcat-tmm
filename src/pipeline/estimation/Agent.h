#pragma once

#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include <nlohmann/json.hpp>

#include "pgm/EvidenceSet.h"
#include "pipeline/estimation/Estimator.h"

namespace tomcat {
    namespace model {

        /**
         * Represents a generic agent for real-time estimations. The agent is
         * responsible for translating the estimations to a proper
         * machine-readable format that can be published to a message bus and
         * consumed by other agents.
         */
        class Agent {
          public:
            //------------------------------------------------------------------
            // Constructors & Destructor
            //------------------------------------------------------------------

            /**
             * Creates an agent with a given ID.
             *
             * @param id: agent's ID
             * @param estimates_topic: message topic where estimates must be
             * published to
             * @param log_topic: message topic where processing log must be
             * published to
             */
            Agent(const std::string& id,
                  const std::string& estimates_topic,
                  const std::string& log_topic);

            virtual ~Agent();

            //------------------------------------------------------------------
            // Copy & Move constructors/assignments
            //------------------------------------------------------------------

            // Copy constructor and assignment should be deleted to avoid
            // implicit slicing and loss of polymorphic behaviour in the
            // subclasses. To deep copy, the clone method must be used if any.
            Agent(const Agent&) = delete;

            Agent& operator=(const Agent&) = delete;

            Agent(Agent&&) = default;

            Agent& operator=(Agent&&) = default;

            //------------------------------------------------------------------
            // Pure virtual functions
            //------------------------------------------------------------------

            /**
             * Converts a message to a proper data format.
             *
             * @param message: message from a message bus
             *
             * @return Data.
             */
            virtual EvidenceSet
            message_to_data(const nlohmann::json& message) = 0;

            /**
             * Builds a message to be published to a message bus containing
             * well formatted estimates computed for a given time step.
             *
             * @param estimators: list of estimators used to compute the
             * estimates
             * @param time_step: time step for which estimates were computed
             *
             * @return Message.
             */
            virtual nlohmann::json estimates_to_message(
                const std::vector<std::shared_ptr<Estimator>>& estimators,
                int time_step) const = 0;

            /**
             * Gets a list of all topics this agent has to subscribe to.
             *
             * @return Set of relevant message topics.
             */
            virtual std::unordered_set<std::string>
            get_topics_to_subscribe() const = 0;

            /**
             * Builds a log message with a given text.
             *
             * @return Log message.
             */
            virtual nlohmann::json
            build_log_message(const std::string& log) const = 0;

            /**
             * Prepare agent to process a new mission.
             */
            virtual void restart() = 0;

            /**
             * Whether the mission timer has reached zero.
             *
             * @return True if mission ended.
             */
            virtual bool is_mission_finished() const = 0;

            //------------------------------------------------------------------
            // Getters & Setters
            //------------------------------------------------------------------
            const std::string& get_id() const;

            const std::string& get_estimates_topic() const;

            const std::string& get_log_topic() const;

          protected:
            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            /**
             * Copy attributes from another agent.
             *
             * @param agent: another agent.
             */
            void copy(const Agent& agent);

            //------------------------------------------------------------------
            // Data members
            //------------------------------------------------------------------
            std::string id;

            std::string estimates_topic;

            std::string log_topic;
        };

    } // namespace model
} // namespace tomcat
