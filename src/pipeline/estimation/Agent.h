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
             * @param version: agent's version
             */
            Agent(const std::string& id, const std::string& version = "");

            virtual ~Agent();

            //------------------------------------------------------------------
            // Copy & Move constructors/assignments
            //------------------------------------------------------------------

            Agent(const Agent&) = default;

            Agent& operator=(const Agent&) = default;

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

            //------------------------------------------------------------------
            // Getters & Setters
            //------------------------------------------------------------------
            const std::string& get_id() const;

            const std::string& get_version() const;

            void set_ignored_observations(
                const std::unordered_set<std::string>& ignored_observations);

            const EstimatorPtrVec& get_estimators() const;

            const nlohmann::json& get_evidence_metadata() const;

          protected:
            //------------------------------------------------------------------
            // Data members
            //------------------------------------------------------------------
            std::string id;

            std::string version;

            EstimatorPtrVec estimators;

            std::unordered_set<std::string> ignored_observations;

            nlohmann::json evidence_metadata;

        };

    } // namespace model
} // namespace tomcat
