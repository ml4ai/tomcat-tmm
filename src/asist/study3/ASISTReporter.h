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
         * Represents a generic reporter for the ASIST program.
         */
        class ASISTReporter : public EstimateReporter {
          public:
            //------------------------------------------------------------------
            // Constructors & Destructor
            //------------------------------------------------------------------

            ASISTReporter() = default;

            ~ASISTReporter() = default;

            //------------------------------------------------------------------
            // Copy & Move constructors/assignments
            //------------------------------------------------------------------
            ASISTReporter(const ASISTReporter& agent);

            ASISTReporter& operator=(const ASISTReporter& agent);

            ASISTReporter(ASISTReporter&&) = default;

            ASISTReporter& operator=(ASISTReporter&&) = default;

          protected:
            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            /**
             * Adds a header section to a json message complying with the ASIST
             * testbed format.
             *
             * @param message: json message
             * @param agent: ASI
             * @param message_type: type of the message. This will go into the
             * type field of the header section
             * @param time_step: time step in which the message is generated. We
             * use this to compute the timestamp of the message relative to the
             * time stamp at the beginning of the mission. By doing it turns the
             * behaviour reproducible.
             */
            void add_header_section(nlohmann::json& message,
                                    const AgentPtr& agent,
                                    const std::string& message_type,
                                    int time_step) const;

            /**
             * Adds a msg section to a json message complying with the ASIST
             * testbed format.
             * @param message: json message
             * @param agent: ASI
             * @param sub_type: type of the message. This will go into the
             * sub_type field of the msg section
             * @param time_step: time step in which the message is generated. We
             * use this to compute the timestamp of the message relative to the
             * time stamp at the beginning of the mission. By doing it turns the
             * behaviour reproducible.
             */
            void add_msg_section(nlohmann::json& message,
                                 const AgentPtr& agent,
                                 const std::string& sub_type,
                                 int time_step) const;

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
        };

    } // namespace model
} // namespace tomcat
