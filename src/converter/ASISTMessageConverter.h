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
         * Converts messages from the TA3 testbed to a format that the model can
         * process.
         */
        class ASISTMessageConverter : public MessageConverter {
          public:
            //------------------------------------------------------------------
            // Constructors & Destructor
            //------------------------------------------------------------------

            /**
             * Creates an instance of the TA3 message converter.
             *
             * @param num_seconds: number of seconds of a mission
             * @param time_step_size: seconds between observations
             * @param map_filepath: path of the map configuration json file
             */
            ASISTMessageConverter(int num_seconds,
                                  int time_step_size,
                                  const std::string& map_filepath);

            ~ASISTMessageConverter();

            //------------------------------------------------------------------
            // Copy & Move constructors/assignments
            //------------------------------------------------------------------
            ASISTMessageConverter(const ASISTMessageConverter& converter);

            ASISTMessageConverter&
            operator=(const ASISTMessageConverter& converter);

            ASISTMessageConverter(ASISTMessageConverter&&) = default;

            ASISTMessageConverter& operator=(ASISTMessageConverter&&) = default;

            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            EvidenceSet
            get_data_from_message(const nlohmann::json& json_message,
                                  nlohmann::json& json_mission_log) override;

            bool is_valid_message_file(
                const boost::filesystem::directory_entry& file) const override;

            /**
             * Gets message topics used to data extraction.
             *
             * @return Message topics.
             */
            std::unordered_set<std::string>
            get_used_topics() const;

            //------------------------------------------------------------------
            // Getters & Setters
            //------------------------------------------------------------------
            std::time_t get_mission_initial_timestamp() const;

            int get_mission_trial_number() const;

          protected:
            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            std::map<std::string, nlohmann::json>
            filter(const std::string& messages_filepath) const override;

          private:
            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            /**
             * Loads map of area configuration as a hash map to easily determine
             * if an area is a room or not by its id.
             *
             * @param map_filepath: path of the map configuration json file
             */
            void load_map_area_configuration(const std::string& map_filepath);

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

            /**
             * Returns observation related to victim saving.
             *
             * @param json_message: json message containing a victim saving
             * event.
             *
             * @return Victim saving event observation.
             */
            void
            fill_victim_saving_observation(const nlohmann::json& json_message);

            /**
             * Returns observation related to being in a room or not.
             *
             * @param json_message: json message containing information about
             * the room the player is in.
             *
             * @return Room observation.
             */
            void fill_room_observation(const nlohmann::json& json_message);

            /**
             * Returns observation about the beep event.
             *
             * @param json_message: json message containing information about
             * the beep played.
             *
             * @return Observation regarding the beep playing.
             */
            void fill_beep_observation(const nlohmann::json& json_message);

            //------------------------------------------------------------------
            // Data members
            //------------------------------------------------------------------

            // Indicates whether a message informing about the mission start was
            // received. Messages received before the mission starts will be
            // ignored.
            bool mission_started = false;
            time_t mission_initial_timestamp;
            int mission_trial_number = -1;

            Tensor3 training_condition;
            Tensor3 area;
            Tensor3 task;
            Tensor3 beep;

            int elapsed_time_steps = 0;

            // Stores the id of all possible areas in the map along with a flag
            // indicating whether the area is a room or not (e.g, yard, hallway
            // etc.).
            std::unordered_map<std::string, bool> map_area_configuration;
        };

    } // namespace model
} // namespace tomcat
