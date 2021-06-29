#pragma once

#include <memory>
#include <string>
#include <vector>

#include "pipeline/estimation/Agent.h"

namespace tomcat {
    namespace model {

        /**
         * Represents a TMM agent for the ASIST program.
         */
        class ASISTAgent : public Agent {
          public:
            //------------------------------------------------------------------
            // Constructors & Destructor
            //------------------------------------------------------------------

            /**
             * Creates an ASIST agent with a given ID.
             *
             * @param id: agent's ID
             * @param estimates_topic: message topic where estimates must be
             * published to
             * @param log_topic: message topic where processing log must be
             * published to
             */
            ASISTAgent(const std::string& id,
                       const std::string& estimates_topic,
                       const std::string& log_topic);

            virtual ~ASISTAgent();

            //------------------------------------------------------------------
            // Copy & Move constructors/assignments
            //------------------------------------------------------------------
            ASISTAgent(const ASISTAgent& agent);

            ASISTAgent& operator=(const ASISTAgent& agent);

            ASISTAgent(ASISTAgent&&) = default;

            ASISTAgent& operator=(ASISTAgent&&) = default;

            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------
            std::vector<nlohmann::json>
            estimates_to_message(int time_step) const override;

            nlohmann::json
            build_log_message(const std::string& log) const override;

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

            //------------------------------------------------------------------
            // Pure virtual functions
            //------------------------------------------------------------------

            /**
             * Fills the header section of a message to be publish
             *
             * @return Header section
             */
            virtual nlohmann::json get_header_section() const = 0;

            /**
             * Fills the msg section section of a message to be publish.
             *
             * @param data_point: index of the data point if estimates were
             * computed for multiple set of observations
             *
             * @return Msg section
             */
            virtual nlohmann::json get_msg_section(int data_point) const = 0;

            /**
             * Fills the data section of a message to be publish.
             *
             * @param time_step: time step to look for estimates
             * @param data_point: index of the data point if estimates were
             * computed for multiple set of observations
             *
             * @return Data section
             */
            virtual nlohmann::json get_data_section(int time_step,
                                                    int data_point) const = 0;
        };

    } // namespace model
} // namespace tomcat
