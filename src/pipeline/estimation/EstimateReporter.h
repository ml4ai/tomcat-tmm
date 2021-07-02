#pragma once

#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include <nlohmann/json.hpp>

#include "pipeline/estimation/Agent.h"

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
                                 int time_step) const = 0;

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
