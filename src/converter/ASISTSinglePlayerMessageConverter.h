#pragma once

#include <iostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <boost/filesystem.hpp>
#include <eigen3/Eigen/Dense>
#include <nlohmann/json.hpp>

#include "ASISTMessageConverter.h"

#include "utils/Definitions.h"
#include "utils/Tensor3.h"

namespace tomcat {
    namespace model {

        /**
         * Converts messages from the ASIST testbed for single-player
         * missions to a format that the model can process.
         */
        class ASISTSinglePlayerMessageConverter : public ASISTMessageConverter {
          public:
            //------------------------------------------------------------------
            // Constructors & Destructor
            //------------------------------------------------------------------

            /**
             * Creates an instance of the message converter.
             *
             * @param num_seconds: number of seconds of a mission
             * @param time_step_size: seconds between observations
             * @param map_filepath: path of the map configuration json file
             */
            ASISTSinglePlayerMessageConverter(int num_seconds,
                                              int time_step_size,
                                              const std::string& map_filepath);

            ~ASISTSinglePlayerMessageConverter();

            //------------------------------------------------------------------
            // Copy & Move constructors/assignments
            //------------------------------------------------------------------
            ASISTSinglePlayerMessageConverter(
                const ASISTSinglePlayerMessageConverter& converter);

            ASISTSinglePlayerMessageConverter&
            operator=(const ASISTSinglePlayerMessageConverter& converter);

            ASISTSinglePlayerMessageConverter(
                ASISTSinglePlayerMessageConverter&&) = default;

            ASISTSinglePlayerMessageConverter&
            operator=(ASISTSinglePlayerMessageConverter&&) = default;

            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            EvidenceSet
            get_data_from_message(const nlohmann::json& json_message,
                                  nlohmann::json& json_mission_log) override;

            bool is_valid_message_file(
                const boost::filesystem::directory_entry& file) const override;

            std::unordered_set<std::string> get_used_topics() const override;

          protected:
            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            /**
             * Copies attributes from another converter.
             *
             * @param converter: original converter
             */
            void
            copy_converter(const ASISTSinglePlayerMessageConverter& converter);

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

            Tensor3 training_condition;
            Tensor3 area;
            Tensor3 task;
            Tensor3 beep;

            int elapsed_time = 0;

            // Stores the id of all possible areas in the map along with a flag
            // indicating whether the area is a room or not (e.g, yard, hallway
            // etc.).
            std::unordered_map<std::string, bool> map_area_configuration;
        };

    } // namespace model
} // namespace tomcat