#pragma once

#include <iostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <eigen3/Eigen/Dense>
#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include "asist/ASISTMessageConverter.h"
#include "utils/Definitions.h"
#include "utils/Tensor3.h"

namespace tomcat {
    namespace model {

        /**
         * Converts Study 3 messages from the ASIST testbed.
         */
        class ASISTStudy3MessageConverter : public ASISTMessageConverter {
          public:
            //------------------------------------------------------------------
            // Structs
            //------------------------------------------------------------------
            struct Labels {
                inline const static std::string ENCOURAGEMENT = "encouragement";
                inline const static std::string LAST_PLACED_MARKERS =
                    "last_placed_markers";
                inline const static std::string REMOVED_MARKERS =
                    "removed_markers";
                inline const static std::string LOCATION_CHANGES =
                    "location_changes";
                inline const static std::string PLAYER_POSITIONS =
                    "player_position";
                inline const static std::string VICTIM_INTERACTIONS =
                    "victim_interaction";
                inline const static std::string SPOKEN_MARKERS =
                    "spoken_marker";
            };

            struct MarkerTypeTexts {
                inline const static std::string NO_VICTIM = "no victim";
                inline const static std::string VICTIM_A = "A";
                inline const static std::string VICTIM_B = "B";
                inline const static std::string VICTIM_C = "critical victim";
                inline const static std::string REGULAR_VICTIM =
                    "regular victim";
                inline const static std::string SOS = "sos";
                inline const static std::string RUBBLE = "rubble";
                inline const static std::string THREAT_ROOM = "threat";
            };

            enum MarkerType {
                NONE,
                NO_VICTIM,
                VICTIM_A,
                VICTIM_B,
                VICTIM_C,
                REGULAR_VICTIM,
                SOS,
                RUBBLE,
                THREAT_ROOM
            };

            inline const static std::unordered_map<MarkerType, std::string>
                MARKER_TYPE_TO_TEXT = {
                    {MarkerType::NO_VICTIM, MarkerTypeTexts::NO_VICTIM},
                    {MarkerType::VICTIM_A, MarkerTypeTexts::VICTIM_A},
                    {MarkerType::VICTIM_B, MarkerTypeTexts::VICTIM_B},
                    {MarkerType::VICTIM_C, MarkerTypeTexts::VICTIM_C},
                    {MarkerType::REGULAR_VICTIM,
                     MarkerTypeTexts::REGULAR_VICTIM},
                    {MarkerType::SOS, MarkerTypeTexts::SOS},
                    {MarkerType::RUBBLE, MarkerTypeTexts::RUBBLE},
                    {MarkerType::THREAT_ROOM, MarkerTypeTexts::THREAT_ROOM}};

            inline const static std::unordered_map<std::string, MarkerType>
                MARKER_TEXT_TO_TYPE = {
                    {MarkerTypeTexts::NO_VICTIM, MarkerType::NO_VICTIM},
                    {MarkerTypeTexts::VICTIM_A, MarkerType::VICTIM_A},
                    {MarkerTypeTexts::VICTIM_B, MarkerType::VICTIM_B},
                    {MarkerTypeTexts::VICTIM_C, MarkerType::VICTIM_C},
                    {MarkerTypeTexts::REGULAR_VICTIM,
                     MarkerType::REGULAR_VICTIM},
                    {MarkerTypeTexts::SOS, MarkerType::SOS},
                    {MarkerTypeTexts::RUBBLE, MarkerType::RUBBLE},
                    {MarkerTypeTexts::THREAT_ROOM, MarkerType::THREAT_ROOM}};

            struct Position {
                double x;
                double z;

                Position() : x(0), z(0) {}

                Position(double x, double z) : x(x), z(z) {}

                explicit Position(const nlohmann::json& serialized_position) {
                    this->x = serialized_position["x"];
                    this->z = serialized_position["z"];
                }

                double distance_to(const Position& pos) const {
                    return sqrt(this->x * pos.x + this->z * pos.z);
                }

                nlohmann::json serialize() const {
                    nlohmann::json json_marker;
                    json_marker["x"] = this->x;
                    json_marker["z"] = this->z;
                    return json_marker;
                }

                bool operator==(const Position& position) const {
                    return position.x == this->x and position.z == this->z;
                }
            };

            struct Marker {
                ASISTStudy3MessageConverter::MarkerType type;
                Position position;

                Marker()
                    : type(ASISTStudy3MessageConverter::MarkerType::NONE),
                      position(Position()) {}

                Marker(ASISTStudy3MessageConverter::MarkerType type,
                       const Position& position)
                    : type(type), position(position) {}

                explicit Marker(const nlohmann::json& serialized_marker) {
                    this->type =
                        MARKER_TEXT_TO_TYPE.at(serialized_marker["type"]);
                    this->position = Position(serialized_marker["position"]);
                }

                bool is_none() const {
                    return this->type ==
                           ASISTStudy3MessageConverter::MarkerType::NONE;
                }

                nlohmann::json serialize() const {
                    nlohmann::json json_marker;
                    if (!this->is_none()) {
                        json_marker["type"] =
                            MARKER_TYPE_TO_TEXT.at(this->type);
                        json_marker["position"] = this->position.serialize();
                    }
                    return json_marker;
                }

                bool operator==(const Marker& marker) const {
                    return marker.position == this->position and
                           marker.type == this->type;
                }
            };

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
            ASISTStudy3MessageConverter(int num_seconds,
                                        int time_step_size,
                                        const std::string& map_filepath,
                                        int num_players);

            ~ASISTStudy3MessageConverter() = default;

            //------------------------------------------------------------------
            // Copy & Move constructors/assignments
            //------------------------------------------------------------------
            ASISTStudy3MessageConverter(
                const ASISTStudy3MessageConverter& converter);

            ASISTStudy3MessageConverter&
            operator=(const ASISTStudy3MessageConverter& converter);

            ASISTStudy3MessageConverter(ASISTStudy3MessageConverter&&) =
                default;

            ASISTStudy3MessageConverter&
            operator=(ASISTStudy3MessageConverter&&) = default;

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
            void copy_converter(const ASISTStudy3MessageConverter& converter);

            EvidenceSet parse_before_mission_start(
                const nlohmann::json& json_message,
                nlohmann::json& json_mission_log) override;

            EvidenceSet parse_after_mission_start(
                const nlohmann::json& json_message,
                nlohmann::json& json_mission_log) override;

            void fill_observation(const nlohmann::json& json_message,
                                  nlohmann::json& json_mission_log) override;

            void prepare_for_new_mission() override;

            void do_offline_conversion_extra_validations() const override;

            void parse_individual_message(
                const nlohmann::json& json_message) override;

          private:
            struct Player {
              public:
                // Default constructor just to allow initial expansion of a
                // vector of players.
                Player() {}

                Player(const std::string& id, const std::string& color)
                    : id(id) {
                    this->color = color;
                    boost::algorithm::to_lower(this->color);

                    if (this->color == "red") {
                        this->index = 0;
                    }
                    else if (this->color == "green") {
                        this->index = 1;
                    }
                    else if (this->color == "blue") {
                        this->index = 2;
                    }
                    else {
                        throw TomcatModelException(fmt::format(
                            "Invalid player color {}.", this->color));
                    }
                }

                std::string id = "";
                std::string color = "";
                std::string role = "";
                int index = -1;
            };

            //------------------------------------------------------------------
            // Static functions
            //------------------------------------------------------------------

            /**
             * Converts a textual representation of a marker type to an instance
             * of marker type without the associated player color.
             *
             * @param textual_type: player color and marker type in textual
             * format
             *
             * @return Marker type
             */
            static ASISTStudy3MessageConverter::MarkerType
            marker_text_to_type(const std::string& textual_type);

            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            /**
             * Loads map of area configuration as a hash map to easily determine
             * if an area is a room or not by its id.
             *
             * @param map_filepath: path of the map configuration json file
             */
            void parse_map(const std::string& map_filepath);

            void parse_players(const nlohmann::json& json_client_info);

            void parse_utterance_message(const nlohmann::json& json_message);

            void
            parse_role_selection_message(const nlohmann::json& json_message,
                                         nlohmann::json& json_mission_log);

            void
            parse_marker_placed_message(const nlohmann::json& json_message);

            void
            parse_marker_removed_message(const nlohmann::json& json_message);

            void
            parse_player_position_message(const nlohmann::json& json_message);

            void parse_new_location_message(const nlohmann::json& json_message);

            void
            parse_victim_placement_message(const nlohmann::json& json_message);

            void
            parse_victim_pickedup_message(const nlohmann::json& json_message);

            void
            parse_victim_triage_message(const nlohmann::json& json_message);

            void
            parse_victim_proximity_message(const nlohmann::json& json_message);

            /**
             * Gets the observations accumulated so far and creates an evidence
             * set with them.
             *
             * @return Evidence set.
             */
            EvidenceSet collect_time_step_evidence();

            /**
             * Writes any remaining information to the log at the end of a
             * mission.
             *
             * @param json_log: json containing extra information about data
             * conversion
             */
            void
            write_to_log_on_mission_finished(nlohmann::json& json_log) const;

            /**
             * Stores the current scoreboard.
             *
             * @param json_message: json containing the scoreboard information
             */
            void parse_scoreboard_message(const nlohmann::json& json_message);

            /**
             * Stores live player info
             *
             * @param json_message: message
             * @param player_number: player number
             */
            void parse_player_state_message(const nlohmann::json& json_message,
                                            int player_number);

            /**
             * Find player object by its ID.
             *
             * @param player_id: player id
             *
             * @return
             */
            Player& get_player_by_id(const std::string& player_id);

            //------------------------------------------------------------------
            // Data members
            //------------------------------------------------------------------

            int num_players;
            int next_time_step;
            int num_players_with_role;
            bool first_mission = true;

            std::unordered_map<std::string, int> player_id_to_index;
            std::vector<Player> players;

            // Data
            int num_encouragement_utterances;

            std::vector<std::vector<Marker>> placed_markers;
            std::vector<std::vector<Marker>> removed_markers;
            std::vector<Position> player_positions;
            std::vector<bool> location_changes;
            std::vector<bool> victim_interactions;
            std::vector<std::unordered_set<MarkerType>> spoken_markers;
        };

    } // namespace model
} // namespace tomcat
