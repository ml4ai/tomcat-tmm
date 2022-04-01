#include "ASISTStudy3MessageConverter.h"

#include <iomanip>

#include "utils/JSONChecker.h"

namespace tomcat {
    namespace model {

        using namespace std;
        namespace alg = boost::algorithm;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        ASISTStudy3MessageConverter::ASISTStudy3MessageConverter(
            int num_seconds,
            int time_step_size,
            const std::string& map_filepath,
            int num_players)
            : ASISTMessageConverter(num_seconds, time_step_size),
              num_players(num_players), next_time_step(0),
              num_players_with_role(0), num_encouragement_utterances(0) {

            this->players.resize(num_players);
            this->parse_map(map_filepath);
        }

        //----------------------------------------------------------------------
        // Copy & Move constructors/assignments
        //----------------------------------------------------------------------
        ASISTStudy3MessageConverter::ASISTStudy3MessageConverter(
            const ASISTStudy3MessageConverter& converter)
            : ASISTMessageConverter(converter.time_steps *
                                        converter.time_step_size,
                                    converter.time_step_size),
              num_players(converter.num_players),
              next_time_step(converter.next_time_step),
              num_players_with_role(converter.num_players_with_role),
              num_encouragement_utterances(
                  converter.num_encouragement_utterances) {
            this->copy_converter(converter);
        }

        ASISTStudy3MessageConverter& ASISTStudy3MessageConverter::operator=(
            const ASISTStudy3MessageConverter& converter) {
            this->copy_converter(converter);
            return *this;
        }

        //----------------------------------------------------------------------
        // Static functions
        //----------------------------------------------------------------------

        ASISTStudy3MessageConverter::MarkerType
        ASISTStudy3MessageConverter::marker_text_to_type(
            const string& textual_type) {
            string type_no_color =
                textual_type.substr(textual_type.find('_') + 1);

            MarkerType type;

            if (type_no_color == "regularvictim") {
                type = MarkerType::REGULAR_VICTIM;
            }
            else if (type_no_color == "criticalvictim") {
                type = MarkerType::VICTIM_C;
            }
            else if (type_no_color == "novictim") {
                type = MarkerType::NO_VICTIM;
            }
            else if (type_no_color == "threat") {
                type = MarkerType::THREAT_ROOM;
            }
            else if (type_no_color == "bonedamage") {
                type = MarkerType::VICTIM_B;
            }
            else if (type_no_color == "abrasion") {
                type = MarkerType::VICTIM_A;
            }
            else if (type_no_color == "sos") {
                type = MarkerType::SOS;
            }
            else if (type_no_color == "rubble") {
                type = MarkerType::RUBBLE;
            }
            else {
                throw TomcatModelException(
                    fmt::format("Invalid marker type {}.", textual_type));
            }

            return type;
        }

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        void ASISTStudy3MessageConverter::copy_converter(
            const ASISTStudy3MessageConverter& converter) {
            ASISTMessageConverter::copy_converter(converter);

            this->num_players = converter.num_players;
            this->next_time_step = converter.next_time_step;
            this->num_players_with_role = converter.num_players_with_role;
            this->num_encouragement_utterances =
                converter.num_encouragement_utterances;
        }

        void ASISTStudy3MessageConverter::prepare_for_new_mission() {
            this->next_time_step = 0;
            this->num_players_with_role = 0;
            this->player_id_to_index.clear();

            this->num_encouragement_utterances = 0;
            this->placed_markers = vector<vector<Marker>>(this->num_players);
            this->removed_markers = vector<vector<Marker>>(this->num_players);
            this->player_positions = vector<Position>(this->num_players);
            this->location_changes = vector<bool>(this->num_players, false);
            this->victim_interactions = vector<bool>(this->num_players, false);
            this->spoken_markers =
                vector<unordered_set<MarkerType>>(this->num_players);
        }

        void
        ASISTStudy3MessageConverter::parse_map(const string& map_filepath) {

            fstream map_file;
            map_file.open(map_filepath);
            if (map_file.is_open()) {
                nlohmann::json json_map = nlohmann::json::parse(map_file);
            }
        }

        unordered_set<string>
        ASISTStudy3MessageConverter::get_used_topics() const {
            unordered_set<string> topics;

            topics.insert("trial");
            topics.insert("observations/events/mission");
            topics.insert("observations/state");
            topics.insert("agent/dialog");
            topics.insert("observations/events/player/role_selected");
            topics.insert("observations/events/mission/planning");
            topics.insert("observations/events/player/marker_placed");
            topics.insert("observations/events/player/marker_removed");
            topics.insert("observations/events/player/location");
            topics.insert("observations/events/player/victim_placed");
            topics.insert("observations/events/player/victim_picked_up");
            topics.insert("observations/events/player/triage");
            topics.insert("observations/events/player/proximity_block");

            return topics;
        }

        EvidenceSet ASISTStudy3MessageConverter::parse_before_mission_start(
            const nlohmann::json& json_message,
            nlohmann::json& json_mission_log) {

            EvidenceSet data;
            if (is_message_of(json_message, "trial")) {
                const string& sub_type = json_message["msg"]["sub_type"];

                if (boost::iequals(sub_type, "start")) {
                    check_field(json_message["data"], "trial_number");
                    check_field(json_message["msg"], "experiment_id");
                    check_field(json_message["msg"], "trial_id");
                    check_field(json_message["data"], "name");
                    check_field(json_message["data"], "map_block_filename");
                    check_field(json_message["data"], "client_info");

                    const string& name = json_message["data"]["name"];

                    json_mission_log["trial"] =
                        json_message["data"]["trial_number"];
                    json_mission_log["experiment_id"] =
                        json_message["msg"]["experiment_id"];
                    json_mission_log["trial_id"] =
                        json_message["msg"]["trial_id"];
                    json_mission_log["team"] = name.substr(0, name.find('_'));

                    const string& map_filename =
                        json_message["data"]["map_block_filename"];
                    if (map_filename.find("SaturnA") != string::npos) {
                        this->first_mission = true;
                        json_mission_log["mission_order"] = 1;
                    }
                    else if (map_filename.find("SaturnB") != string::npos) {
                        this->first_mission = false;
                        json_mission_log["mission_order"] = 2;
                        // experiment_mission.
                        // Saturn A and C are mission 1
                        // Saturn B and D are mission 2
                    }
                    else {
                        throw TomcatModelException(
                            fmt::format("Could identify mission order. "
                                        "Message.data.map_block_filename is "
                                        "{}. It does not contain "
                                        "either SaturnA or SaturnB.",
                                        map_filename));
                    }

                    this->parse_players(json_message["data"]["client_info"]);
                }
                else if (boost::iequals(sub_type, "stop")) {
                    this->mission_finished = true;
                }
            }
            else if (is_message_of(
                         json_message, "event", "Event:MissionState")) {
                check_field(json_message["data"], "mission_state");

                const string& mission_state =
                    json_message["data"]["mission_state"];
                if (boost::iequals(mission_state, "start")) {
                    this->mission_started = true;
                    this->elapsed_seconds = -1;
                }
                else if (boost::iequals(mission_state, "stop")) {
                    this->mission_finished = true;
                }
            }
            else if (!this->mission_finished) {
                // Everything else we might be interested in storing before the
                // mission begins.
                this->fill_observation(json_message, json_mission_log);
            }

            return data;
        }

        EvidenceSet ASISTStudy3MessageConverter::parse_after_mission_start(
            const nlohmann::json& json_message,
            nlohmann::json& json_mission_log) {

            EvidenceSet data;

            if (is_message_of(json_message, "observation", "state")) {
                check_field(json_message["data"], "mission_timer");

                const string& timer = json_message["data"]["mission_timer"];
                int elapsed_seconds = this->get_elapsed_time(timer);

                if (elapsed_seconds >= 0 && this->elapsed_seconds < 0) {
                    // t = 0 (First time step). We store the timestamp when the
                    // mission really started.
                    const string& timestamp = json_message["msg"]["timestamp"];
                    json_mission_log["initial_timestamp"] = timestamp;
                    json_mission_log["initial_elapsed_milliseconds"] =
                        json_message["data"]["elapsed_milliseconds"];

                    data = this->collect_time_step_evidence();
                    this->elapsed_seconds = 0;
                }

                // Collect evidence when there's a time step transition
                for (int t = this->elapsed_seconds + this->time_step_size;
                     t <= elapsed_seconds;
                     t++) {
                    data.hstack(this->collect_time_step_evidence());

                    this->elapsed_seconds += this->time_step_size;
                    if (this->elapsed_seconds >=
                        this->time_steps * this->time_step_size - 1) {
                        this->write_to_log_on_mission_finished(
                            json_mission_log);
                        this->mission_finished = true;
                        break;
                    }
                }
            }
            else if (is_message_of(
                         json_message, "event", "Event:MissionState")) {
                check_field(json_message["data"]["mission_state"],
                            "mission_state");

                const string& mission_state =
                    json_message["data"]["mission_state"];
                if (boost::iequals(mission_state, "stop")) {
                    this->write_to_log_on_mission_finished(json_mission_log);
                    this->mission_finished = true;
                }
            }

            if (!this->mission_finished) {
                this->fill_observation(json_message, json_mission_log);
            }

            return data;
        }

        void ASISTStudy3MessageConverter::parse_players(
            const nlohmann::json& json_client_info) {

            for (const auto& info_per_player : json_client_info) {
                check_field(info_per_player, "participant_id");
                check_field(info_per_player, "callsign");

                Player player((string)info_per_player["participant_id"],
                              (string)info_per_player["callsign"]);

                this->players[player.index] = player;
                this->player_id_to_index[player.id] = player.index;
            }
        }

        void ASISTStudy3MessageConverter::fill_observation(
            const nlohmann::json& json_message,
            nlohmann::json& json_mission_log) {

            if (is_message_of(json_message, "event", "Event:RoleSelected")) {
                this->parse_role_selection_message(json_message,
                                                   json_mission_log);
            }

            if (this->mission_started) {
                if (is_message_of(
                        json_message, "event", "Event:dialogue_event")) {
                    this->parse_utterance_message(json_message);
                }
                else if (is_message_of(
                             json_message, "event", "Event:MarkerPlaced")) {
                    this->parse_marker_placed_message(json_message);
                }
                else if (is_message_of(
                             json_message, "event", "Event:MarkerRemoved")) {
                    this->parse_marker_removed_message(json_message);
                }
                else if (is_message_of(
                             json_message, "event", "Event:location")) {
                    this->parse_new_location_message(json_message);
                }
                else if (is_message_of(json_message, "observation", "state")) {
                    this->parse_player_position_message(json_message);
                }
                else if (is_message_of(
                             json_message, "event", "Event:VictimPlaced")) {
                    this->parse_victim_placement_message(json_message);
                }
                else if (is_message_of(
                             json_message, "event", "Event:VictimPickedUp")) {
                    this->parse_victim_pickedup_message(json_message);
                }
                else if (is_message_of(json_message, "event", "Event:Triage")) {
                    this->parse_victim_triage_message(json_message);
                }
                else if (is_message_of(json_message,
                                       "event",
                                       "Event:ProximityBlockInteraction")) {
                    this->parse_victim_proximity_message(json_message);
                }
            }
        }

        //
        //            string player_id;
        //            int player_number = -1;
        //
        //            if (EXISTS("participant_id", json_message["data"])) {
        //                if (json_message["data"]["participant_id"] == nullptr)
        //                {
        //                    if (json_message["msg"]["sub_type"] != "FoV") {
        //                        throw TomcatModelException("Participant ID is
        //                        null.");
        //                    }
        //                    // If it's FoV, we just ignore FoV Messages.
        //                    return;
        //                }
        //                player_id = json_message["data"]["participant_id"];
        //                if (EXISTS(player_id, this->player_id_to_number)) {
        //                    player_number =
        //                    this->player_id_to_number[player_id];
        //                }
        //            }
        //
        //            if (player_id != "") {
        //                // Player evidence
        //                if (json_message["header"]["message_type"] == "event"
        //                &&
        //                    json_message["msg"]["sub_type"] ==
        //                    "Event:RoleSelected") {
        //                    this->parse_role_selection_message(json_message,
        //                                                       player_number);
        //                }
        //                else if (json_message["header"]["message_type"] ==
        //                             "observation" &&
        //                         json_message["msg"]["sub_type"] == "state") {
        //                    this->parse_player_state_message(json_message,
        //                                                     player_number);
        //                }
        //            }
        //        }

        void ASISTStudy3MessageConverter::parse_utterance_message(
            const nlohmann::json& json_message) {
            check_field(json_message["data"], "extractions");

            for (const auto& json_extraction :
                 json_message["data"]["extractions"]) {
                check_field(json_extraction, "labels");

                for (const auto& label : json_extraction["labels"]) {
                    if (boost::iequals((string)label, "Encouragement")) {
                        this->num_encouragement_utterances++;
                    }
                }
            }
        }

        void ASISTStudy3MessageConverter::parse_role_selection_message(
            const nlohmann::json& json_message,
            nlohmann::json& json_mission_log) {
            check_field(json_message["data"], "new_role");
            check_field(json_message["data"], "participant_id");

            const string& new_role = json_message["data"]["new_role"];
            Player& player = this->get_player_by_id(
                (string)json_message["data"]["participant_id"]);

            if (player.role.empty()) {
                this->num_players_with_role++;
            }

            if (boost::iequals(new_role, "transport_specialist")) {
                player.role = "transporter";
            }
            else if (boost::iequals(new_role, "engineering_specialist")) {
                player.role = "engineer";
            }
            else if (boost::iequals(new_role, "medical_specialist")) {
                player.role = "medic";
            }
            else {
                throw TomcatModelException(
                    fmt::format("Role {} invalid.", new_role));
            }

            if (this->num_players_with_role == this->num_players) {
                json_mission_log["players"] = nlohmann::json::array();

                for (const auto& p : this->players) {
                    nlohmann::json json_player;
                    json_player["id"] = p.id;
                    json_player["color"] = p.color;
                    json_player["role"] = p.role;
                    json_mission_log["players"].push_back(json_player);
                }
            }
        }

        void ASISTStudy3MessageConverter::parse_marker_placed_message(
            const nlohmann::json& json_message) {
            check_field(json_message["data"], "marker_x");
            check_field(json_message["data"], "marker_z");
            check_field(json_message["data"], "participant_id");
            check_field(json_message["data"], "type");

            MarkerType type =
                marker_text_to_type((string)json_message["data"]["type"]);
            Position pos((double)json_message["data"]["marker_x"],
                         (double)json_message["data"]["marker_z"]);
            Marker marker(type, pos);

            int player_order = this->player_id_to_index.at(
                (string)json_message["data"]["participant_id"]);

            this->placed_markers[player_order].push_back(marker);
        }

        void ASISTStudy3MessageConverter::parse_marker_removed_message(
            const nlohmann::json& json_message) {
            check_field(json_message["data"], "marker_x");
            check_field(json_message["data"], "marker_z");
            check_field(json_message["data"], "participant_id");
            check_field(json_message["data"], "type");

            MarkerType type =
                marker_text_to_type((string)json_message["data"]["type"]);
            Position pos((double)json_message["data"]["marker_x"],
                         (double)json_message["data"]["marker_z"]);
            Marker marker(type, pos);

            // If the marker was just placed, remove it from the list of markers
            // placed instead.
            for (int i = 0; i < this->num_players; i++) {
                for (int j = 0; j < this->placed_markers[i].size(); j++) {
                    if (marker == this->placed_markers[i][j]) {
                        this->placed_markers[i].erase(
                            this->placed_markers[i].begin() + j);
                        return;
                    }
                }
            }

            int player_order = this->player_id_to_index.at(
                (string)json_message["data"]["participant_id"]);

            this->removed_markers[player_order].push_back(marker);
        }

        void ASISTStudy3MessageConverter::parse_player_position_message(
            const nlohmann::json& json_message) {
            check_field(json_message["data"], "x");
            check_field(json_message["data"], "z");
            check_field(json_message["data"], "participant_id");

            Position pos((double)json_message["data"]["x"],
                         (double)json_message["data"]["z"]);
            int player_order = this->player_id_to_index.at(
                (string)json_message["data"]["participant_id"]);

            this->player_positions[player_order] = pos;
        }

        void ASISTStudy3MessageConverter::parse_new_location_message(
            const nlohmann::json& json_message) {
            check_field(json_message["data"], "participant_id");

            int player_order = this->player_id_to_index.at(
                (string)json_message["data"]["participant_id"]);

            this->location_changes[player_order] = true;
        }

        void ASISTStudy3MessageConverter::parse_victim_placement_message(
            const nlohmann::json& json_message) {
            check_field(json_message["data"], "participant_id");

            int player_order = this->player_id_to_index.at(
                (string)json_message["data"]["participant_id"]);

            this->victim_interactions[player_order] = true;
        }

        void ASISTStudy3MessageConverter::parse_victim_pickedup_message(
            const nlohmann::json& json_message) {
            check_field(json_message["data"], "participant_id");

            int player_order = this->player_id_to_index.at(
                (string)json_message["data"]["participant_id"]);

            this->victim_interactions[player_order] = true;
        }

        void ASISTStudy3MessageConverter::parse_victim_triage_message(
            const nlohmann::json& json_message) {
            check_field(json_message["data"], "participant_id");

            int player_order = this->player_id_to_index.at(
                (string)json_message["data"]["participant_id"]);

            this->victim_interactions[player_order] = true;
        }

        void ASISTStudy3MessageConverter::parse_victim_proximity_message(
            const nlohmann::json& json_message) {
            check_field(json_message["data"], "participant_id");

            int player_order = this->player_id_to_index.at(
                (string)json_message["data"]["participant_id"]);

            this->victim_interactions[player_order] = true;
        }

        void ASISTStudy3MessageConverter::parse_scoreboard_message(
            const nlohmann::json& json_message) {
            //            int score =
            //            json_message["data"]["scoreboard"]["TeamScore"];
            //            this->team_score(0, 0, 0) = score;
        }

        void ASISTStudy3MessageConverter::parse_player_state_message(
            const nlohmann::json& json_message, int player_number) {
            //            int x = json_message["data"]["x"];
            //            int z = json_message["data"]["z"];
            //            this->player_position[player_number] = Position(x, z);
        }

        EvidenceSet ASISTStudy3MessageConverter::collect_time_step_evidence() {
            nlohmann::json dict_data;

            dict_data[Labels::ENCOURAGEMENT] =
                this->num_encouragement_utterances;
            this->num_encouragement_utterances = 0;

            nlohmann::json json_last_placed_markers = nlohmann::json::array();
            nlohmann::json json_removed_markers = nlohmann::json::array();
            nlohmann::json json_location_changes = nlohmann::json::array();
            nlohmann::json json_victim_interactions = nlohmann::json::array();
            nlohmann::json json_player_positions = nlohmann::json::array();
            nlohmann::json json_spoken_markers = nlohmann::json::array();
            for (int player_order = 0; player_order < this->num_players;
                 player_order++) {
                // Placed markers
                if (!this->placed_markers[player_order].empty()) {
                    // Just save the last one placed
                    json_last_placed_markers.push_back(
                        this->placed_markers[player_order].back().serialize());
                }
                else {
                    json_last_placed_markers.push_back(nlohmann::json());
                }
                this->placed_markers[player_order].clear();

                // Removed markers
                nlohmann::json json_markers = nlohmann::json::array();
                for (const auto& marker : this->removed_markers[player_order]) {
                    json_markers.push_back(marker.serialize());
                }
                json_removed_markers.push_back(json_markers);
                this->removed_markers[player_order].clear();

                // Location changes
                json_location_changes.push_back(
                    (bool)this->location_changes[player_order]);
                this->location_changes[player_order] = false;

                // Victim interactions
                json_victim_interactions.push_back(
                    (bool)this->victim_interactions[player_order]);
                this->victim_interactions[player_order] = false;

                // Player positions
                json_player_positions.push_back(
                    this->player_positions[player_order].serialize());

                // Spoken markers
                nlohmann::json json_marker_types = nlohmann::json::array();
                for (const auto& marker_type :
                     this->spoken_markers[player_order]) {
                    json_marker_types.push_back(
                        MARKER_TYPE_TO_TEXT.at(marker_type));
                }
                json_spoken_markers.push_back(json_marker_types);
                this->spoken_markers[player_order].clear();
            }

            dict_data[Labels::LAST_PLACED_MARKERS] = json_last_placed_markers;
            dict_data[Labels::REMOVED_MARKERS] = json_removed_markers;
            dict_data[Labels::LOCATION_CHANGES] = json_location_changes;
            dict_data[Labels::VICTIM_INTERACTIONS] = json_victim_interactions;
            dict_data[Labels::PLAYER_POSITIONS] = json_player_positions;
            dict_data[Labels::SPOKEN_MARKERS] = json_spoken_markers;

            // Clear data that should not persist across time steps

            vector<vector<nlohmann::json>> dict_data_vec(1);
            dict_data_vec[0].push_back(dict_data);
            EvidenceSet data(dict_data_vec);

            this->next_time_step += 1;
            return data;
        }

        bool ASISTStudy3MessageConverter::is_valid_message_file(
            const boost::filesystem::directory_entry& file) const {
            const string filename = file.path().filename().string();

            return filename.find("TrialMessages") != string::npos &&
                   filename.find("Training") == string::npos &&
                   filename.find("PlanningASR") == string::npos &&
                   filename.find("FoV") == string::npos &&
                   file.path().extension().string() == ".metadata";
        }

        void
        ASISTStudy3MessageConverter::do_offline_conversion_extra_validations()
            const {
            if (this->mission_started &&
                this->players.size() < this->num_players) {
                throw TomcatModelException(
                    fmt::format("Only {} players joined the mission.",
                                this->players.size()));
            }
        }

        void ASISTStudy3MessageConverter::parse_individual_message(
            const nlohmann::json& json_message) {}

        void ASISTStudy3MessageConverter::write_to_log_on_mission_finished(
            nlohmann::json& json_log) const {}

        ASISTStudy3MessageConverter::Player&
        ASISTStudy3MessageConverter::get_player_by_id(const string& player_id) {
            int player_index = this->player_id_to_index[player_id];
            return this->players[player_index];
        }

    } // namespace model
} // namespace tomcat
