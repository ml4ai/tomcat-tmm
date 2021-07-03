#pragma once

#include <memory>

#include "pgm/EvidenceSet.h"
#include "pipeline/estimation/Agent.h"
#include "pipeline/estimation/EstimateReporter.h"
#include "utils/Definitions.h"

namespace tomcat {
    namespace model {

        /**
         * Generic estimation process for a DBN model.
         */
        class EstimationProcess {
          public:
            //------------------------------------------------------------------
            // Types, Enums & Constants
            //------------------------------------------------------------------

            //------------------------------------------------------------------
            // Constructors & Destructor
            //------------------------------------------------------------------

            /**
             * Creates a generic estimation process for a model.
             *
             * @param agent: agent used in the estimation
             * @param reporter: class responsible for reporting estimates
             * computed during the process
             *
             */
            EstimationProcess(const AgentPtr& agent,
                              const EstimateReporterPtr& reporter);

            virtual ~EstimationProcess();

            //------------------------------------------------------------------
            // Copy & Move constructors/assignments
            //------------------------------------------------------------------

            // Copy constructor and assignment should be deleted to avoid
            // implicit slicing and loss of polymorphic behaviour in the
            // subclasses. To deep copy, the clone method must be used.
            EstimationProcess(const EstimationProcess&) = delete;

            EstimationProcess& operator=(const EstimationProcess&) = delete;

            EstimationProcess(EstimationProcess&&) = default;

            EstimationProcess& operator=(EstimationProcess&&) = default;

            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            /**
             * Assigns training data to the estimator. Some estimators (mainly
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
             * Computes estimations for all agents in the estimation process.
             *
             * @param observations: new observations
             */
            virtual void estimate(const EvidenceSet& test_data);

            //------------------------------------------------------------------
            // Pure virtual functions
            //------------------------------------------------------------------

            /**
             * Publishes last computed estimates to an external resource that
             * depends on a concrete estimation process.
             */
            virtual void publish_last_estimates() = 0;

            //------------------------------------------------------------------
            // Getters & Setters
            //------------------------------------------------------------------

            void set_display_estimates(bool display_estimates);

            const AgentPtr& get_agent() const;

          protected:
            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            /**
             * Copies data members from another estimation process.
             */
            void copy_estimation(const EstimationProcess& estimation);

            //------------------------------------------------------------------
            // Data members
            //------------------------------------------------------------------
            AgentPtr agent;

            // Whether the estimates computed by the agents must be displayed
            // in the evaluation file.
            bool display_estimates = false;

            // Number of time steps the estimation already processed.
            int last_time_step;

            EstimateReporterPtr reporter;
        };

    } // namespace model
} // namespace tomcat
