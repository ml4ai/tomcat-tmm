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
         * Converts messages from the ASIST testbed for multi-player
         * missions to a format that the model can process.
         */
        class ASISTMultiPlayerMessageConverter : public ASISTMessageConverter {
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
             * @param num_players: number of players in the mission
             */
            ASISTMultiPlayerMessageConverter(int num_seconds,
                                             int time_step_size,
                                             const std::string& map_filepath,
                                             int num_players);

            ~ASISTMultiPlayerMessageConverter();

            //------------------------------------------------------------------
            // Copy & Move constructors/assignments
            //------------------------------------------------------------------
            ASISTMultiPlayerMessageConverter(
                const ASISTMultiPlayerMessageConverter& converter);

            ASISTMultiPlayerMessageConverter&
            operator=(const ASISTMultiPlayerMessageConverter& converter);

            ASISTMultiPlayerMessageConverter(
                ASISTMultiPlayerMessageConverter&&) = default;

            ASISTMultiPlayerMessageConverter&
            operator=(ASISTMultiPlayerMessageConverter&&) = default;

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
            copy_converter(const ASISTMultiPlayerMessageConverter& converter);

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
             * Parse message before mission starts.
             *
             * @param json_message: json message.
             * @param json_mission_log: includes info to be put in the
             * conversion metadata file.
             *
             * @return Data collected from the parsed message.
             */
            EvidenceSet
            parse_before_mission_start(const nlohmann::json& json_message,
                                       nlohmann::json& json_mission_log);

            /**
             * Parse message before mission starts.
             *
             * @param json_message: json message.
             * @param json_mission_log: includes info to be put in the
             * conversion metadata file.
             *
             * @return Data collected from the parsed message.
             */
            EvidenceSet
            parse_after_mission_start(const nlohmann::json& json_message,
                                      nlohmann::json& json_mission_log);

            /**
             * Parse message and fill the appropriate observation tensor.
             *
             * @param json_message: json message with a particular observation.
             */
            void fill_observation(const nlohmann::json& json_message);

            //------------------------------------------------------------------
            // Data members
            //------------------------------------------------------------------

            // Indicates whether a message informing about the mission start was
            // received. Messages received before the mission starts will be
            // ignored.
            bool mission_started = false;
            int elapsed_time = 0;
            int num_players;

            // IDs are sequential numbers starting from zero and indicate the
            // position in the vector of observations.
            std::unordered_map<std::string, int> player_name_to_id;

            // Observations
            std::vector<Tensor3> task_per_player;
            std::vector<Tensor3> role_per_player;
        };

    } // namespace model
} // namespace tomcat
