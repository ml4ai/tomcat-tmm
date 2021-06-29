#pragma once

#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include <nlohmann/json.hpp>

#include "converter/MessageConverter.h"
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
            // Member functions
            //------------------------------------------------------------------

            /**
             * Assigns training data to the estimators. Some estimators (mainly
             * the baseline ones) might calculate estimates based on training
             * data rather than test data.
             *
             * @param training_data: training data
             */
            void set_training_data(const EvidenceSet& training_data);

            /**
             * Aks estimators to save the estimates computed. In a cross
             * validation process, this method will be called multiple times. In
             * the end, we will have a list of estimates calculated in each one
             * of the cross validation steps.
             */
            void keep_estimates();

            /**
             * Clear cumulative estimates computed by the estimators.
             */
            void clear_estimates();

            /**
             * Computes estimates given new observations.
             *
             * @param observations: new observations
             */
            void estimate(const EvidenceSet& observations);

            /**
             * Adds a new estimator to the agent.
             *
             * @param estimator: Estimator
             */
            void add_estimator(const EstimatorPtr& estimator);

            /**
             * Gets a list of all topics this agent has to subscribe to.
             *
             * @return Set of relevant message topics.
             */
            std::unordered_set<std::string> get_topics_to_subscribe() const;

            //------------------------------------------------------------------
            // Virtual functions
            //------------------------------------------------------------------

            /**
             * Prepares the estimation process to start again.
             */
            virtual void prepare();

            /**
             * Writes information about the estimation in a json object.
             *
             * @param json: json object
             */
            virtual void get_info(nlohmann::json& json) const;

            /**
             * Builds a message to be published to a message bus containing
             * well formatted estimates computed for a given time step.
             *
             * @param time_step: time step for which estimates were computed
             *
             * @return Messages.
             */
            virtual std::vector<nlohmann::json>
            estimates_to_message(int time_step) const;

            /**
             * Builds a log message with a given text.
             *
             * @return Log message.
             */
            virtual nlohmann::json
            build_log_message(const std::string& log) const;

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

            EstimatorPtrVec estimators;

            std::unordered_set<std::string> ignored_observations;

            nlohmann::json evidence_metadata;
        };

    } // namespace model
} // namespace tomcat
