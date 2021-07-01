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
            // Observable node names
            inline const static std::string ROLE = "Role";
            inline const static std::string TASK = "Task";
            inline const static std::string AREA = "Area";
            inline const static std::string SECTION = "Section";
            inline const static std::string SEEN_MARKER = "SeenMarker";
            inline const static std::string MARKER_LEGEND = "MarkerLegend";
            inline const static std::string MAP_INFO = "MapInfo";
            inline const static std::string MARKER_LEGEND_ASSIGNMENT = "MarkerLegendAssignment";
            inline const static std::string MAP_INFO_ASSIGNMENT = "MapInfoAssignment";

            // Task values
            const static int NO_TASK = 0;
            const static int CARRYING_VICTIM = 1;
            const static int CLEARING_RUBBLE = 2;
            const static int SAVING_REGULAR = 3;
            const static int SAVING_CRITICAL = 4;

            // Role values
            const static int NO_ROLE = 0;
            const static int SEARCH = 1;
            const static int HAMMER = 2;
            const static int MEDICAL = 3;

            // Area
            const static int HALLWAY = 0;
            const static int ROOM = 1;

            // Building sections
            const static int OUT_OF_BUILDING = 0;

            // Map info
            const static int SECTIONS_2N4 = 0;
            const static int SECTIONS_3N4 = 1;
            const static int SECTIONS_6N4 = 2;

            // Marker legend
            const static int MARKER_LEGEND_A = 0;
            const static int MARKER_LEGEND_B = 1;

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

            EvidenceSet parse_before_mission_start(
                const nlohmann::json& json_message,
                nlohmann::json& json_mission_log) override;

            EvidenceSet parse_after_mission_start(
                const nlohmann::json& json_message,
                nlohmann::json& json_mission_log) override;

            void fill_observation(const nlohmann::json& json_message) override;

            void prepare_for_new_mission() override;

            void do_offline_conversion_extra_validations() const override;

          private:
            //------------------------------------------------------------------
            // Structs
            //------------------------------------------------------------------

            struct BoundingBox {
                int x1;
                int x2;
                int z1;
                int z2;

                BoundingBox(int x1, int x2, int z1, int z2)
                    : x1(x1), x2(x2), z1(z1), z2(z2) {}
            };

            struct Position {
                int x;
                int z;

                Position(int x, int z) : x(x), z(z) {}

                bool is_inside(const BoundingBox& box) const {
                    return this->x <= box.x2 && this->x >= box.x1 &&
                           this->z <= box.z2 && this->z >= box.z1;
                }
            };

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
             * Adds a player to the list of detected players.
             *
             * @param player_name: name of the player
             */
            void add_player(const std::string& player_name);

            /**
             * Extracts the trial number from a string containing the trial
             * number as a suffix.
             *
             * @param textual_trial_number: trial number in the format:
             * T000<number>
             *
             * @return Numeric trial number
             */
            int get_numeric_trial_number(
                const std::string& textual_trial_number) const;

            /**
             * Fills map info and marker legend for each one of the
             * participants in the trial.
             *
             * @param json_client_info: json object containing information
             * about the kind of map and marker legend received per player
             *
             */
            void fill_client_info_data(const nlohmann::json& json_client_info);

            /**
             * Gets the observations accumulated so far and creates an evidence
             * set with them.
             *
             * @return Evidence set.
             */
            EvidenceSet build_evidence_set_from_observations();

            /**
             * Find the section of the building the player is in.
             *
             * @param player_id: player id
             *
             * @return Number of the section
             */
            int get_building_section(int player_id) const;

            //------------------------------------------------------------------
            // Data members
            //------------------------------------------------------------------

            int num_players;

            // Stores the id of all possible areas in the map along with a flag
            // indicating whether the area is a room or not (e.g, yard, hallway
            // etc.).
            std::unordered_map<std::string, bool> map_area_configuration;

            std::vector<BoundingBox> building_sections;

            // IDs are sequential numbers starting from zero and indicate the
            // position in the vector of observations.
            std::unordered_map<std::string, int> player_name_to_id;

            // Observations
            std::vector<Tensor3> task_per_player;
            std::vector<Tensor3> role_per_player;
            std::vector<Tensor3> area_per_player; // Depends on location data
            std::vector<Tensor3> seen_marker_per_player; // Depends on FoV data
            std::vector<Tensor3> section_per_player;
            std::vector<Tensor3> marker_legend_per_player;
            std::vector<Tensor3> map_info_per_player;

            std::vector<Position> player_position;
        };

    } // namespace model
} // namespace tomcat
