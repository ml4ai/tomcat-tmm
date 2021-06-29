#pragma once

#include <iostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <boost/filesystem.hpp>
#include <eigen3/Eigen/Dense>
#include <nlohmann/json.hpp>

#include "MessageConverter.h"

#include "utils/Definitions.h"
#include "utils/Tensor3.h"

namespace tomcat {
    namespace model {

        /**
         * Converts messages from the ASIST testbed to a format that the model
         * can process.
         */
        class ASISTMessageConverter : public MessageConverter {
          public:
            //------------------------------------------------------------------
            // Constructors & Destructor
            //------------------------------------------------------------------

            /**
             * Creates an instance of the message converter.
             *
             * @param num_seconds: number of seconds of a mission
             * @param time_step_size: seconds between observations
             */
            ASISTMessageConverter(int num_seconds, int time_step_size);

            virtual ~ASISTMessageConverter();

            //------------------------------------------------------------------
            // Copy & Move constructors/assignments
            //------------------------------------------------------------------
            ASISTMessageConverter(const ASISTMessageConverter&) = delete;

            ASISTMessageConverter&
            operator=(const ASISTMessageConverter&) = delete;

            ASISTMessageConverter(ASISTMessageConverter&&) = default;

            ASISTMessageConverter& operator=(ASISTMessageConverter&&) = default;

            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            EvidenceSet
            get_data_from_message(const nlohmann::json& json_message,
                                  nlohmann::json& json_mission_log) override;

            //------------------------------------------------------------------
            // Getters & Setters
            //------------------------------------------------------------------
            std::time_t get_mission_initial_timestamp() const;

            int get_mission_trial_number() const;

            const std::string& get_experiment_id() const;

          protected:
            //------------------------------------------------------------------
            // Pure virtual functions
            //------------------------------------------------------------------
            /**
             * Parse message before mission starts.
             *
             * @param json_message: json message.
             * @param json_mission_log: includes info to be put in the
             * conversion metadata file.
             *
             * @return Data collected from the parsed message.
             */
            virtual EvidenceSet
            parse_before_mission_start(const nlohmann::json& json_message,
                                       nlohmann::json& json_mission_log) = 0;

            /**
             * Parse message before mission starts.
             *
             * @param json_message: json message.
             * @param json_mission_log: includes info to be put in the
             * conversion metadata file.
             *
             * @return Data collected from the parsed message.
             */
            virtual EvidenceSet
            parse_after_mission_start(const nlohmann::json& json_message,
                                      nlohmann::json& json_mission_log) = 0;

            /**
             * Parse message and fill the appropriate observation tensor.
             *
             * @param json_message: json message with a particular observation.
             */
            virtual void
            fill_observation(const nlohmann::json& json_message) = 0;

            /**
             * Clea nup before new mission starts
             */
            virtual void prepare_for_new_mission() = 0;

            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            std::map<std::string, nlohmann::json>
            filter(const std::string& messages_filepath) const override;

            /**
             * Copies attributes from another converter.
             *
             * @param converter: original converter
             */
            void copy_converter(const ASISTMessageConverter& converter);

            /**
             * Converts a string with remaining minutes and seconds to the total
             * number of seconds elapsed since the mission started.
             *
             * @param time: string containing the remaining time formatted as
             * mm : ss
             *
             * @return Elapsed time in seconds.
             */
            int get_elapsed_time(const std::string& time);

            //------------------------------------------------------------------
            // Data members
            //------------------------------------------------------------------
            // Indicates whether a message informing about the mission start was
            // received. Messages received before the mission starts will be
            // ignored.
            bool mission_started = false;

            int elapsed_time = 0;

            time_t mission_initial_timestamp;

            int mission_trial_number = -1;

            std::string experiment_id;
        };

    } // namespace model
} // namespace tomcat
