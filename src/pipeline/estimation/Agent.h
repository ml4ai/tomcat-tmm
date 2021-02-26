#pragma once

#include <memory>
#include <string>
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
             */
            Agent(const std::string& id);

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

            //------------------------------------------------------------------
            // Getters & Setters
            //------------------------------------------------------------------
            const std::string& get_id() const;

          protected:
            //------------------------------------------------------------------
            // Data members
            //------------------------------------------------------------------
            std::string id;
        };

    } // namespace model
} // namespace tomcat
