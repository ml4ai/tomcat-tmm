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
            inline const static std::string MARKER_LEGEND_ASSIGNMENT_LABEL =
                "MarkerLegendVersionAssignment";
            inline const static std::string PLAYER_MARKER_LEGEND_VERSION_LABEL =
                "PlayerMarkerLegendVersion";
            inline const static std::string OTHER_PLAYER_NEARBY_MARKER =
                "OtherPlayerNearbyMarker";
            inline const static std::string PLAYER_ROLE_LABEL =
                "ObservedPlayerRole";
            inline const static std::string PLAYER_TASK_LABEL = "PlayerTask";
            inline const static std::string PLAYER_AREA_LABEL = "PlayerArea";
            inline const static std::string PLAYER_PLACED_MARKER_LABEL =
                "PlayerPlacedMarker";
            inline const static std::string FINAL_TEAM_SCORE_LABEL =
                "FinalTeamScore";
            inline const static std::string TEAM_SCORE_LABEL = "TeamScore";
            inline const static std::string MAP_VERSION_ASSIGNMENT_LABEL =
                "MapVersionAssignment";
            inline const static std::string PLAYER_MAP_VERSION_LABEL =
                "PlayerMapVersion";
            inline const static std::string OBS_PLAYER_BUILDING_SECTION_LABEL =
                "ObservedPlayerBuildingSection";

            // Task values
            const static int NO_TASK = 0;
            const static int SAVING_REGULAR = 1;
            const static int SAVING_CRITICAL = 2;
            const static int CLEARING_RUBBLE = 3;
            const static int CARRYING_VICTIM = 4;

            // Role values
            const static int SEARCH = 0;
            const static int HAMMER = 1;
            const static int MEDICAL = 2;

            // Area
            const static int HALLWAY = 0;
            const static int ROOM = 1;

            // Building sections
            const static int OUT_OF_BUILDING = 0;

            // Map info
            const static int SECTIONS_2N4 = 0;
            const static int SECTIONS_3N4 = 1;
            const static int SECTIONS_6N4 = 2;

            // Markers
            const static int MARKER_LEGEND_A = 0;
            const static int MARKER_LEGEND_B = 1;

            const static int NO_MARKER_PLACED = 0;

            // Bounding box of the main part of the map. Used to split the map
            // into 6 sections. Staging area is not included.
            int MAP_SECTION_MIN_X = -2225;
            int MAP_SECTION_MAX_X = -2087;
            int MAP_SECTION_MIN_Z = -10;
            int MAP_SECTION_MAX_Z = 60;

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

            void parse_individual_message(
                const nlohmann::json& json_message) override;

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

                bool equals(const Position& other_position) const {
                    return this->x == other_position.x &&
                           this->z == other_position.z;
                }

                double get_distance(const Position& other_position) const {
                    return sqrt(pow(this->x - other_position.x, 2) +
                                pow(this->z - other_position.z, 2));
                }
            };

            struct Player {
                std::string id;
                std::string callsign; // red, blue or green
                std::string unique_id;
                std::string name;
            };

            struct MarkerBlock {
                std::string player_id;
                Position position;
                int number;

                MarkerBlock(const Position& position) : position(position) {}

                bool overwrites(const MarkerBlock& other_block) const {
                    return this->position.equals(other_block.position);
                }
            };

            struct Door {
                std::string id;
                Position position;

                Door(const Position& position) : position(position) {}
            };

            struct MarkerBlockAndDoor {
                MarkerBlock block;
                Door door;

                MarkerBlockAndDoor(MarkerBlock block, Door door)
                    : block(block), door(door) {}
            };

            //------------------------------------------------------------------
            // Static functions
            //------------------------------------------------------------------

            /**
             * Determines whether two block positions are within 2-blocks range
             * as determines by study 2.
             *
             *             Block
             *       Block Block Block
             * Block Block   ?   Block Block
             *       Block Block Block
             *             Block
             *
             * @param pos1: Position of the first block/entity
             * @param pos2: Position of the second block/entity
             *
             * @return True if they are close enough
             */
            static bool
            are_within_marker_block_detection_radius(const Position& pos1,
                                                     const Position& pos2);

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
             * Adds a player to the list of players in the mission.
             *
             * @param player: player
             */
            void add_player(const Player& player);

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
             * @param json_log: json containing extra information about data
             * conversion
             *
             */
            void fill_players(const nlohmann::json& json_client_info,
                              nlohmann::json& json_log);

            /**
             * Store observations that do not change over time.
             */
            void fill_fixed_measures();

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

            /**
             * Get the closest door to a certain position.
             *
             * @param position: position to be compared to the door position
             *
             * @return Closest door
             */
            Door get_closest_door(const Position& position) const;

            /**
             * Writes any remaining information to the log at the end of a
             * mission.
             *
             * @param json_log: json containing extra information about data
             * conversion
             */
            void
            write_to_log_on_mission_finished(nlohmann::json& json_log) const;

            //------------------------------------------------------------------
            // Data members
            //------------------------------------------------------------------

            int num_players;

            // Stores the id of all possible areas in the map along with a flag
            // indicating whether the area is a room or not (e.g, yard, hallway
            // etc.).
            std::unordered_map<std::string, bool> map_area_configuration;
            std::vector<BoundingBox> building_sections;
            std::vector<Door> doors;

            // Numbers are sequential numbers starting from zero and indicate
            // the position in the vector of observations. Id's and names are
            // stored because the testbed is not stable yet and some messages
            // are addressed by the player's ids, whereas others are identified
            // by their names.
            std::unordered_map<std::string, int> player_id_to_number;
            std::unordered_map<std::string, int> player_name_to_number;
            std::vector<Player> players;

            // Observations
            std::vector<Tensor3> task_per_player;
            std::vector<Tensor3> role_per_player;
            std::vector<Tensor3> area_per_player;
            std::vector<Tensor3> section_per_player;
            std::vector<Tensor3> marker_legend_per_player;
            std::vector<Tensor3> map_info_per_player;
            std::vector<Tensor3> player_placed_marker;
            int final_score;
            int map_version_assignment;
            int marker_legend_version_assignment;
            int current_team_score;

            // Auxiliary variables that change over the course of the game
            std::vector<Position> player_position;
            std::vector<MarkerBlock> placed_marker_blocks;

            // When processing offline, some measures will be available to
            // process and we save them as ground truth observations so we can
            // evaluate the quality of the inferences made by the model
            nlohmann::json json_measures;

            // Variables used for report generation

            // Information about marker blocks seen, closest door per player and
            // time step.
            std::vector<
                std::unordered_map<int, std::vector<MarkerBlockAndDoor>>>
                nearby_markers_info;
            int next_time_step;
        };

    } // namespace model
} // namespace tomcat
