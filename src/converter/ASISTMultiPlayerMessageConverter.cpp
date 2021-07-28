#include "ASISTMultiPlayerMessageConverter.h"

#include <fstream>
#include <iomanip>

#include <boost/algorithm/string.hpp>

namespace tomcat {
    namespace model {

        using namespace std;
        namespace alg = boost::algorithm;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        ASISTMultiPlayerMessageConverter::ASISTMultiPlayerMessageConverter(
            int num_seconds,
            int time_step_size,
            const std::string& map_filepath,
            int num_players)
            : ASISTMessageConverter(num_seconds, time_step_size),
              num_players(num_players) {

            this->load_map_area_configuration(map_filepath);
        }

        ASISTMultiPlayerMessageConverter::~ASISTMultiPlayerMessageConverter() {}

        //----------------------------------------------------------------------
        // Copy & Move constructors/assignments
        //----------------------------------------------------------------------
        ASISTMultiPlayerMessageConverter::ASISTMultiPlayerMessageConverter(
            const ASISTMultiPlayerMessageConverter& converter)
            : ASISTMessageConverter(converter.time_steps *
                                        converter.time_step_size,
                                    converter.time_step_size) {
            this->copy_converter(converter);
        }

        ASISTMultiPlayerMessageConverter&
        ASISTMultiPlayerMessageConverter::operator=(
            const ASISTMultiPlayerMessageConverter& converter) {
            this->copy_converter(converter);
            return *this;
        }

        //----------------------------------------------------------------------
        // Static functions
        //----------------------------------------------------------------------
        bool ASISTMultiPlayerMessageConverter::
            are_within_marker_block_detection_radius(const Position& pos1,
                                                     const Position& pos2) {

            return pos1.get_distance(pos2) <= 2;

            //            auto square_around =
            //                BoundingBox(pos1.x - 1, pos1.x + 1, pos1.z - 1,
            //                pos1.z + 1);
            //            if (pos2.is_inside(square_around)) {
            //                return true;
            //            }
            //            else {
            //                // Tips of the star
            //                if ((pos1.x - 2 == pos2.x || pos1.x + 2 == pos2.x)
            //                &&
            //                    pos1.z == pos2.z) {
            //                    return true;
            //                }
            //                else if ((pos1.z - 2 == pos2.z || pos1.z + 2 ==
            //                pos2.z) &&
            //                         pos1.x == pos2.x) {
            //                    return true;
            //                }
            //            }
            //
            //            return false;
        }

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        void ASISTMultiPlayerMessageConverter::copy_converter(
            const ASISTMultiPlayerMessageConverter& converter) {
            ASISTMessageConverter::copy_converter(converter);
            this->mission_started = converter.mission_started;
            this->elapsed_time = converter.elapsed_time;
            this->num_players = converter.num_players;
            this->player_id_to_number = converter.player_id_to_number;
            this->player_name_to_number = converter.player_name_to_number;
            this->players = converter.players;
            this->task_per_player = converter.task_per_player;
            this->role_per_player = converter.role_per_player;
            this->area_per_player = converter.area_per_player;
            this->section_per_player = converter.section_per_player;
            this->marker_legend_per_player = converter.marker_legend_per_player;
            this->map_info_per_player = converter.map_info_per_player;
            this->final_score = converter.final_score;
            this->map_version_assignment = converter.map_version_assignment;
            this->marker_legend_version_assignment =
                converter.marker_legend_version_assignment;
            this->player_position = converter.player_position;
            this->placed_marker_blocks = converter.placed_marker_blocks;
            this->nearby_markers_info = converter.nearby_markers_info;
            this->placed_marker_per_player = converter.placed_marker_per_player;
            this->victim_in_fov_per_player = converter.victim_in_fov_per_player;
            this->agreement_speech_per_player =
                converter.agreement_speech_per_player;
            this->marker_legend_speech_per_player =
                converter.marker_legend_speech_per_player;
            this->action_enter_speech_per_player =
                converter.action_enter_speech_per_player;
        }

        void ASISTMultiPlayerMessageConverter::load_map_area_configuration(
            const string& map_filepath) {

            this->map_area_configuration.clear();
            this->doors.clear();

            fstream map_file;
            map_file.open(map_filepath);
            if (map_file.is_open()) {
                nlohmann::json json_map = nlohmann::json::parse(map_file);

                this->fill_building_sections();

                // Hallway or Room
                for (const auto& location : json_map["locations"]) {
                    const string& area_type = location["type"];
                    const string& area_id = location["id"];

                    this->map_area_configuration[area_id] =
                        area_type.find("room") != string::npos;
                }

                // Doors
                for (const auto& location : json_map["connections"]) {
                    const string& area_type = location["type"];
                    const string& area_id = location["id"];

                    if (area_type == "door") {
                        // Include each block that comprises a door as a
                        // door object in the list because the position to a
                        // doorway (in terms of blocks) will affect how
                        // marker blocks are detected.
                        int x = location["bounds"]["coordinates"][0]["x"];
                        int z = location["bounds"]["coordinates"][0]["z"];
                        Door door({x, z});
                        door.id = area_id;
                        this->doors.push_back(door);

                        x = location["bounds"]["coordinates"][1]["x"];
                        z = location["bounds"]["coordinates"][1]["z"];
                        door.position = Position(x, z);
                        this->doors.push_back(door);
                    }
                }
            }
            else {
                stringstream ss;
                ss << "Map configuration file in " << map_filepath
                   << " does not exist.";
                throw TomcatModelException(ss.str());
            }
        }

        void ASISTMultiPlayerMessageConverter::fill_building_sections() {
            this->building_sections.clear();
            this->expanded_building_sections.clear();

            // Split the map into 6 non-overlapping sections
            int section_width = (MAP_SECTION_MAX_X - MAP_SECTION_MIN_X) / 3;
            int section_height = (MAP_SECTION_MAX_Z - MAP_SECTION_MIN_Z) / 2;

            BoundingBox section1(MAP_SECTION_MIN_X,
                                 MAP_SECTION_MIN_X + section_width,
                                 MAP_SECTION_MIN_Z,
                                 MAP_SECTION_MIN_Z + section_height);
            BoundingBox section2(MAP_SECTION_MIN_X,
                                 MAP_SECTION_MIN_X + section_width,
                                 MAP_SECTION_MIN_Z + section_height,
                                 MAP_SECTION_MAX_Z);
            BoundingBox section3(MAP_SECTION_MIN_X + section_width,
                                 MAP_SECTION_MIN_X + 2 * section_width,
                                 MAP_SECTION_MIN_Z,
                                 MAP_SECTION_MIN_Z + section_height);
            BoundingBox section4(MAP_SECTION_MIN_X + section_width,
                                 MAP_SECTION_MIN_X + 2 * section_width,
                                 MAP_SECTION_MIN_Z + section_height,
                                 MAP_SECTION_MAX_Z);
            BoundingBox section5(MAP_SECTION_MIN_X + 2 * section_width,
                                 MAP_SECTION_MAX_X,
                                 MAP_SECTION_MIN_Z,
                                 MAP_SECTION_MIN_Z + section_height);
            BoundingBox section6(MAP_SECTION_MIN_X + 2 * section_width,
                                 MAP_SECTION_MAX_X,
                                 MAP_SECTION_MIN_Z + section_height,
                                 MAP_SECTION_MAX_Z);

            this->building_sections.push_back(section1);
            this->building_sections.push_back(section2);
            this->building_sections.push_back(section3);
            this->building_sections.push_back(section4);
            this->building_sections.push_back(section5);
            this->building_sections.push_back(section6);

            auto [upper_section2, section7] = section2.get_horizontal_split();
            auto [left_section3, section8] = section3.get_vertical_split();
            auto [left_section4, section9] = section4.get_vertical_split();
            auto [left_section6, section10] = section6.get_vertical_split();

            this->expanded_building_sections.push_back(section1);
            this->expanded_building_sections.push_back(upper_section2);
            this->expanded_building_sections.push_back(left_section3);
            this->expanded_building_sections.push_back(left_section4);
            this->expanded_building_sections.push_back(left_section6);
            this->expanded_building_sections.push_back(section6);
            this->expanded_building_sections.push_back(section7);
            this->expanded_building_sections.push_back(section8);
            this->expanded_building_sections.push_back(section9);
            this->expanded_building_sections.push_back(section10);
        }

        unordered_set<string>
        ASISTMultiPlayerMessageConverter::get_used_topics() const {
            unordered_set<string> topics;

            topics.insert("trial");
            topics.insert("observations/events/mission");
            topics.insert("observations/state");
            topics.insert("observations/events/player/tool_used");
            topics.insert("observations/events/player/victim_picked_up");
            topics.insert("observations/events/player/victim_placed");
            topics.insert("observations/events/player/role_selected");
            topics.insert("observations/events/player/triage");
            topics.insert("observations/events/scoreboard");
            topics.insert("observations/events/player/location");
            topics.insert("agent/pygl_fov/player/3d/summary");
            topics.insert("agent/measures");
            topics.insert("observations/events/player/marker_placed");
            topics.insert("agent/dialog");
            topics.insert("ground_truth/mission/victims_list");

            return topics;
        }

        EvidenceSet
        ASISTMultiPlayerMessageConverter::parse_before_mission_start(
            const nlohmann::json& json_message,
            nlohmann::json& json_mission_log) {

            EvidenceSet data;
            if (json_message["header"]["message_type"] == "event" &&
                json_message["msg"]["sub_type"] == "Event:MissionState") {
                string mission_state = json_message["data"]["mission_state"];
                alg::to_lower(mission_state);

                if (mission_state == "start") {
                    this->mission_started = true;
                    this->elapsed_time = this->time_step_size;

                    // Store initial timestamp
                    const string& timestamp =
                        json_message["header"]["timestamp"];
                    json_mission_log["initial_timestamp"] = timestamp;

                    tm t{};
                    istringstream ss(timestamp);
                    // The precision of the timestamp will be in seconds.
                    // milliseconds are ignored. This can be reaccessed
                    // later if necessary. The milliseconds could be stored
                    // in a separate attribute of this class.
                    ss >> get_time(&t, "%Y-%m-%dT%T");
                    if (!ss.fail()) {
                        this->mission_initial_timestamp = mktime(&t);
                    }

                    // Store observations in the initial time step
                    data = this->build_evidence_set_from_observations();
                }
                else if (mission_state == "stop") {
                    this->mission_finished = true;
                }
            }
            else if (json_message["header"]["message_type"] == "trial") {
                string sub_type = json_message["msg"]["sub_type"];
                alg::to_lower(sub_type);

                if (sub_type == "start") {
                    this->mission_trial_number = this->get_numeric_trial_number(
                        json_message["data"]["trial_number"]);
                    this->experiment_id = json_message["msg"]["experiment_id"];

                    const string& team_n_trial =
                        json_message["data"]["trial_number"];
                    string team_id =
                        team_n_trial.substr(0, team_n_trial.find("_"));
                    string trial_id =
                        team_n_trial.substr(team_n_trial.find("_") + 1);
                    // Team planning or no planning
                    this->planning_condition =
                        stoi((string)json_message["data"]["condition"]) - 1;

                    json_mission_log["trial_id"] = trial_id;
                    json_mission_log["team_id"] = team_id;
                    json_mission_log["experiment_id"] =
                        json_message["msg"]["experiment_id"];

                    this->experiment_id = json_mission_log["experiment_id"];

                    if (!EXISTS("client_info", json_message["data"])) {
                        throw TomcatModelException(
                            "No information about players.");
                    }

                    this->fill_players(json_message["data"]["client_info"],
                                       json_mission_log);
                    this->fill_fixed_measures();
                }
                else if (sub_type == "stop") {
                    this->mission_finished = true;
                }
            }

            if (!this->mission_finished) {
                this->fill_observation(json_message);
            }

            return data;
        }

        void
        ASISTMultiPlayerMessageConverter::add_player(const Player& player) {
            int player_number = this->player_id_to_number.size();
            this->player_id_to_number[player.id] = player_number;
            if (player.name != "") {
                // Some messages are in a different format and we need to use
                // the player name to retrieve some measurements.
                this->player_name_to_number[player.name] = player_number;
            }
            this->players.push_back(player);

            this->task_per_player.push_back(Tensor3(NO_TASK));
            this->role_per_player.push_back(Tensor3(SEARCH));
            this->area_per_player.push_back(Tensor3(HALLWAY));
            this->section_per_player.push_back(Tensor3(OUT_OF_BUILDING));
            this->expanded_section_per_player.push_back(
                Tensor3(OUT_OF_BUILDING));
            this->marker_legend_per_player.push_back(Tensor3(NO_OBS));
            this->map_info_per_player.push_back(Tensor3(NO_OBS));
            this->player_position.push_back({0, 0});
            this->nearby_markers_info.resize(this->nearby_markers_info.size() +
                                             1);
            this->placed_marker_per_player.push_back(Tensor3(NO_MARKER_PLACED));
            this->victim_in_fov_per_player.push_back(Tensor3(NO_VICTIM_IN_FOV));

            this->agreement_speech_per_player.push_back(Tensor3(NO_SPEECH));
            this->marker_legend_speech_per_player.push_back(Tensor3(NO_SPEECH));
            this->action_enter_speech_per_player.push_back(Tensor3(NO_SPEECH));
        }

        int ASISTMultiPlayerMessageConverter::get_numeric_trial_number(
            const std::string& textual_trial_number) const {
            int trial_number = -1;
            try {
                // Remove first character which is the letter T.
                trial_number = stoi(textual_trial_number.substr(1));
            }
            catch (invalid_argument& exp) {
            }

            return trial_number;
        }

        void ASISTMultiPlayerMessageConverter::fill_players(
            const nlohmann::json& json_client_info, nlohmann::json& json_log) {

            json_log["players"] = nlohmann::json::array();

            for (const auto& info_per_player : json_client_info) {
                Player player;
                player.id = info_per_player["participant_id"];
                player.callsign = info_per_player["callsign"];
                player.unique_id = info_per_player["uniqueid"];

                if (EXISTS("playername", info_per_player)) {
                    player.name = info_per_player["playername"];
                }
                this->add_player(player);

                nlohmann::json json_player;
                json_player["id"] = player.id;
                json_player["callsign"] = player.callsign;
                json_player["uniqueid"] = player.unique_id;
                json_player["name"] = player.name;
                json_log["players"].push_back(json_player);
            }
        }

        void ASISTMultiPlayerMessageConverter::fill_fixed_measures() {
            if (!this->json_measures.empty()) {
                // M1: Team score
                this->final_score = json_measures["data"]["M1"]["final_score"];

                // M3: Map
                /*
                 * The assignments of maps per player are:
                 * Player 1 | Player 2 | Player 3 | Value
                 * 24	  34	     64	        0
                 * 24	  64	     34	        1
                 * 34	  24	     64	        2
                 * 34	  64	     24	        3
                 * 64	  24	     34	        4
                 * 64	  34	     24	        5
                 */
                unordered_set<int> map_assignment = {0, 1, 2, 3, 4, 5};
                for (const nlohmann::json& json_map :
                     json_measures["data"]["M3"]) {

                    int player_number;
                    if (EXISTS("participant_id", json_map)) {
                        player_number = this->player_id_to_number
                                            [json_map["participant_id"]];
                    }
                    else {
                        if (player_name_to_number.empty()) {
                            throw TomcatModelException(
                                "The measures agent requires the player name "
                                "but that was not provided by the trial.");
                        }
                        player_number = this->player_name_to_number
                                            [json_map["player_name"]];
                    }

                    const string& static_map = json_map["map_name"];
                    string version = static_map.substr(static_map.length() - 2);
                    if (version == "24") {
                        this->map_info_per_player[player_number] =
                            Tensor3(SECTIONS_2N4);

                        if (player_number == 0) {
                            map_assignment.erase(2);
                            map_assignment.erase(3);
                            map_assignment.erase(4);
                            map_assignment.erase(5);
                        }
                        else if (player_number == 1) {
                            map_assignment.erase(0);
                            map_assignment.erase(1);
                            map_assignment.erase(3);
                            map_assignment.erase(5);
                        }
                        else if (player_number == 2) {
                            map_assignment.erase(0);
                            map_assignment.erase(1);
                            map_assignment.erase(2);
                            map_assignment.erase(4);
                        }
                    }
                    else if (version == "34") {
                        this->map_info_per_player[player_number] =
                            Tensor3(SECTIONS_3N4);

                        if (player_number == 0) {
                            map_assignment.erase(0);
                            map_assignment.erase(1);
                            map_assignment.erase(4);
                            map_assignment.erase(5);
                        }
                        else if (player_number == 1) {
                            map_assignment.erase(1);
                            map_assignment.erase(2);
                            map_assignment.erase(3);
                            map_assignment.erase(4);
                        }
                        else if (player_number == 2) {
                            map_assignment.erase(0);
                            map_assignment.erase(2);
                            map_assignment.erase(3);
                            map_assignment.erase(5);
                        }
                    }
                    else if (version == "64") {
                        this->map_info_per_player[player_number] =
                            Tensor3(SECTIONS_6N4);

                        if (player_number == 0) {
                            map_assignment.erase(0);
                            map_assignment.erase(1);
                            map_assignment.erase(2);
                            map_assignment.erase(3);
                        }
                        else if (player_number == 1) {
                            map_assignment.erase(0);
                            map_assignment.erase(2);
                            map_assignment.erase(4);
                            map_assignment.erase(5);
                        }
                        else if (player_number == 2) {
                            map_assignment.erase(1);
                            map_assignment.erase(3);
                            map_assignment.erase(4);
                            map_assignment.erase(5);
                        }
                    }
                }
                this->map_version_assignment = *map_assignment.begin();

                // M6: Marker block legend
                for (const nlohmann::json& json_marker :
                     json_measures["data"]["M6"]) {

                    int player_number;
                    if (EXISTS("participant_id", json_marker)) {
                        player_number = this->player_id_to_number
                                            [json_marker["participant_id"]];
                    }
                    else {
                        if (player_name_to_number.empty()) {
                            throw TomcatModelException(
                                "The measures agent requires the player name "
                                "but that was not provided by the trial.");
                        }
                        player_number = this->player_name_to_number
                                            [json_marker["player_name"]];
                    }

                    const string& marker_legend =
                        json_marker["marker_block_legend"];
                    if (marker_legend[0] == 'A') {
                        this->marker_legend_per_player[player_number] =
                            Tensor3(MARKER_LEGEND_A);
                    }
                    else if (marker_legend[0] == 'B') {
                        this->marker_legend_per_player[player_number] =
                            Tensor3(MARKER_LEGEND_B);
                        this->marker_legend_version_assignment = player_number;
                    }
                }
            }
        }

        EvidenceSet ASISTMultiPlayerMessageConverter::
            build_evidence_set_from_observations() {

            EvidenceSet data;

            // Per team
            data.add_data(PLANNING_CONDITION_LABEL,
                          Tensor3(this->planning_condition));

            // Per player
            for (int player_number = 0; player_number < this->players.size();
                 player_number++) {
                data.add_data(get_player_variable_label(PLAYER_TASK_LABEL,
                                                        player_number + 1),
                              this->task_per_player.at(player_number));
                data.add_data(get_player_variable_label(PLAYER_ROLE_LABEL,
                                                        player_number + 1),
                              this->role_per_player.at(player_number));
                data.add_data(get_player_variable_label(PLAYER_AREA_LABEL,
                                                        player_number + 1),
                              this->area_per_player.at(player_number));
                data.add_data(
                    get_player_variable_label(
                        PLAYER_MARKER_LEGEND_VERSION_LABEL, player_number + 1),
                    this->marker_legend_per_player.at(player_number));
                data.add_data(
                    MARKER_LEGEND_ASSIGNMENT_LABEL,
                    Tensor3::constant(
                        1, 1, 1, this->marker_legend_version_assignment));
                data.add_data(
                    TEAM_SCORE_LABEL,
                    Tensor3::constant(1, 1, 1, this->current_team_score));
                data.add_data(
                    get_player_variable_label(PLAYER_PLACED_MARKER_LABEL,
                                              player_number + 1),
                    this->placed_marker_per_player.at(player_number));

                // The player detects a block if it's close enough to a block
                // and this block is close enough to a door.
                int nearby_marker = NO_NEARBY_MARKER;
                for (const auto& block : this->placed_marker_blocks) {
                    if (are_within_marker_block_detection_radius(
                            this->player_position[player_number],
                            block.position) &&
                        block.player_id != this->players[player_number].id &&
                        block.player_id != this->players[player_number].name) {

                        Door door = this->get_closest_door(block.position);
                        if (are_within_marker_block_detection_radius(
                                block.position, door.position)) {
                            nearby_marker = block.number;

                            if (!EXISTS(
                                    this->next_time_step,
                                    this->nearby_markers_info[player_number])) {
                                this->nearby_markers_info
                                    [player_number][this->next_time_step] =
                                    vector<MarkerBlockAndDoor>();
                            }

                            MarkerBlockAndDoor block_and_door(block, door);
                            this
                                ->nearby_markers_info[player_number]
                                                     [this->next_time_step]
                                .push_back(block_and_door);
                        }
                    }
                }
                data.add_data(
                    get_player_variable_label(OTHER_PLAYER_NEARBY_MARKER,
                                              player_number + 1),
                    nearby_marker);

                // Map model
                int section = this->get_building_section(player_number);
                data.add_data(
                    get_player_variable_label(OBS_PLAYER_BUILDING_SECTION_LABEL,
                                              player_number + 1),
                    section);
                section = this->get_expanded_building_section(player_number);
                data.add_data(get_player_variable_label(
                                  OBS_PLAYER_EXPANDED_BUILDING_SECTION_LABEL,
                                  player_number + 1),
                              section);
                data.add_data(get_player_variable_label(
                                  PLAYER_MAP_VERSION_LABEL, player_number + 1),
                              this->map_info_per_player.at(player_number));
                data.add_data(
                    MAP_VERSION_ASSIGNMENT_LABEL,
                    Tensor3::constant(1, 1, 1, this->map_version_assignment));

                // NLP
                data.add_data(get_player_variable_label(PLAYER_AGREEMENT_LABEL,
                                                        player_number + 1),
                              this->agreement_speech_per_player[player_number]);
                data.add_data(
                    get_player_variable_label(
                        PLAYER_MARKER_LEGEND_VERSION_SPEECH_LABEL,
                        player_number + 1),
                    this->marker_legend_speech_per_player[player_number]);
                data.add_data(
                    get_player_variable_label(PLAYER_ACTION_ENTER_SPEECH_LABEL,
                                              player_number + 1),
                    this->action_enter_speech_per_player[player_number]);

                // FoV
                data.add_data(
                    get_player_variable_label(PLAYER_VICTIM_IN_FOV_LABEL,
                                              player_number + 1),
                    this->victim_in_fov_per_player[player_number]);

                // Observations from measures
                if (!json_measures.empty()) {
                    // Final score
                    data.add_data(
                        FINAL_TEAM_SCORE_LABEL,
                        Tensor3::constant(1, 1, 1, this->final_score));
                }

                // Carrying ans saving a victim are tasks that have
                // an explicit end event. We don't reset the last
                // observation for this task then. It's going to be
                // reset when its ending is detected.
                int last_task =
                    this->task_per_player.at(player_number)(0, 0)(0, 0);
                if (last_task == CLEARING_RUBBLE) {
                    this->task_per_player[player_number] = Tensor3(NO_TASK);
                }

                this->placed_marker_per_player[player_number] =
                    Tensor3(NO_MARKER_PLACED);

                this->victim_in_fov_per_player[player_number] =
                    Tensor3(NO_VICTIM_IN_FOV);

                // Reset speeches
                this->agreement_speech_per_player[player_number] =
                    Tensor3(NO_SPEECH);
                this->marker_legend_speech_per_player[player_number] =
                    Tensor3(NO_SPEECH);
                this->action_enter_speech_per_player[player_number] =
                    Tensor3(NO_SPEECH);
            }

            this->next_time_step += 1;

            return data;
        }

        EvidenceSet ASISTMultiPlayerMessageConverter::parse_after_mission_start(
            const nlohmann::json& json_message,
            nlohmann::json& json_mission_log) {

            EvidenceSet data;
            if (json_message["header"]["message_type"] == "observation" &&
                json_message["msg"]["sub_type"] == "state") {
                const string& timer = json_message["data"]["mission_timer"];
                int elapsed_time = this->get_elapsed_time(timer);

                if (elapsed_time == this->elapsed_time + this->time_step_size) {
                    // Every time there's a transition, we store the last
                    // observations collected.

                    data = this->build_evidence_set_from_observations();

                    this->elapsed_time += this->time_step_size;
                    if (this->elapsed_time >=
                        this->time_steps * this->time_step_size) {
                        this->write_to_log_on_mission_finished(
                            json_mission_log);
                        this->mission_finished = true;
                    }
                }
            }
            else if (json_message["header"]["message_type"] == "event" &&
                     json_message["msg"]["sub_type"] == "Event:MissionState" &&
                     json_message["data"]["mission_state"] == "Stop") {
                this->write_to_log_on_mission_finished(json_mission_log);
                this->mission_finished = true;
            }

            if (!this->mission_finished) {
                this->fill_observation(json_message);
            }

            return data;
        }

        void ASISTMultiPlayerMessageConverter::fill_observation(
            const nlohmann::json& json_message) {

            string player_id;
            int player_number = -1;
            if (EXISTS("participant_id", json_message["data"])) {
                player_id = json_message["data"]["participant_id"];
                if (EXISTS(player_id, this->player_id_to_number)) {
                    player_number = this->player_id_to_number[player_id];
                }
            }
            else if (EXISTS("playername", json_message["data"])) {
                player_id = json_message["data"]["playername"];
                if (EXISTS(player_id, this->player_name_to_number)) {
                    player_number = this->player_name_to_number[player_id];
                }
                else if (EXISTS(player_id, this->player_id_to_number)) {
                    // Some FoV data contains the player id in the playername
                    // field.
                    player_number = this->player_id_to_number[player_id];
                }
            }

            if (player_id == "") {
                if (json_message["header"]["message_type"] == "observation" &&
                    json_message["msg"]["sub_type"] == "Event:Scoreboard") {

                    this->current_team_score =
                        json_message["data"]["scoreboard"]["TeamScore"];
                }
            }
            else if (player_number >= 0) {
                // Observations that are individual for each player
                if (json_message["header"]["message_type"] == "event" &&
                    json_message["msg"]["sub_type"] == "Event:ToolUsed") {

                    string block_type =
                        json_message["data"]["target_block_type"];
                    alg::to_lower(block_type);
                    string tool_type = json_message["data"]["tool_type"];
                    alg::to_lower(tool_type);

                    if (block_type.find("gravel") != string::npos &&
                        tool_type == "hammer") {
                        this->task_per_player[player_number] =
                            Tensor3(CLEARING_RUBBLE);
                    }
                }
                else if (json_message["header"]["message_type"] == "event" &&
                         json_message["msg"]["sub_type"] == "Event:Triage") {
                    if (json_message["data"]["triage_state"] == "IN_PROGRESS") {
                        if (EXISTS("type", json_message["data"])) {
                            string victim_type = json_message["data"]["type"];
                            alg::to_lower(victim_type);

                            if (victim_type == "regular") {
                                this->task_per_player[player_number] =
                                    Tensor3(SAVING_REGULAR);
                            }
                            else if (victim_type == "critical") {
                                this->task_per_player[player_number] =
                                    Tensor3(SAVING_CRITICAL);
                            }
                        }
                        else {
                            // Old format
                            string victim_color = json_message["data"]["color"];
                            alg::to_lower(victim_color);

                            if (victim_color == "green") {
                                this->task_per_player[player_number] =
                                    Tensor3(SAVING_REGULAR);
                            }
                            else if (victim_color == "yellow") {
                                this->task_per_player[player_number] =
                                    Tensor3(SAVING_CRITICAL);
                            }
                        }
                    }
                    else {
                        this->task_per_player[player_number] = Tensor3(NO_TASK);

                        if (json_message["data"]["triage_state"] ==
                            "SUCCESSFUL") {
                            stringstream id;
                            id << json_message["data"]["victim_x"] << "#"
                               << json_message["data"]["victim_z"];
                            this->rescued_victims.insert(id.str());
                        }
                    }
                }
                else if (json_message["header"]["message_type"] == "event" &&
                         json_message["msg"]["sub_type"] ==
                             "Event:VictimPickedUp") {
                    this->task_per_player[player_number] =
                        Tensor3(CARRYING_VICTIM);
                }
                else if (json_message["header"]["message_type"] == "event" &&
                         json_message["msg"]["sub_type"] ==
                             "Event:VictimPlaced") {
                    this->task_per_player[player_number] = Tensor3(NO_TASK);
                }
                else if (json_message["header"]["message_type"] == "event" &&
                         json_message["msg"]["sub_type"] ==
                             "Event:RoleSelected") {
                    string role = json_message["data"]["new_role"];
                    alg::to_lower(role);

                    if (role == "search_specialist" || role == "search" ||
                        role == "none") {
                        this->role_per_player[player_number] = Tensor3(SEARCH);
                    }
                    else if (role == "hazardous_material_specialist" ||
                             role == "hammer") {
                        this->role_per_player[player_number] = Tensor3(HAMMER);
                    }
                    else if (role == "medical_specialist" ||
                             role == "medical") {
                        this->role_per_player[player_number] = Tensor3(MEDICAL);
                    }
                    else {
                        stringstream ss;
                        ss << "Invalid role (" << role << ") chosen by player "
                           << player_id;
                        throw TomcatModelException(ss.str());
                    }
                }
                else if (json_message["header"]["message_type"] == "event" &&
                         json_message["msg"]["sub_type"] == "Event:location") {

                    if (EXISTS("locations", json_message["data"])) {
                        bool area = false;
                        for (const auto& location :
                             json_message["data"]["locations"]) {

                            const string& location_id = location["id"];
                            if (location_id == "UNKNOWN") {
                                continue;
                            }

                            if (!EXISTS(location_id,
                                        this->map_area_configuration)) {
                                stringstream ss;
                                ss << "Location id " << location_id
                                   << " does not exist in the map.";
                                throw TomcatModelException(ss.str());
                            }

                            area = this->map_area_configuration.at(location_id);
                            if (area) {
                                break;
                            }
                        }

                        this->area_per_player[player_number] =
                            Tensor3((int)area);
                    }
                }
                else if (json_message["header"]["message_type"] ==
                             "observation" &&
                         json_message["msg"]["sub_type"] == "state") {
                    int x = json_message["data"]["x"];
                    int z = json_message["data"]["z"];
                    this->player_position[player_number] = Position(x, z);
                }
                else if (json_message["header"]["message_type"] == "event" &&
                         json_message["msg"]["sub_type"] ==
                             "Event:MarkerPlaced") {

                    int x = json_message["data"]["marker_x"];
                    int z = json_message["data"]["marker_z"];
                    MarkerBlock marker_block({x, z});
                    marker_block.player_id = player_id;
                    const string& type = json_message["data"]["type"];
                    marker_block.number = stoi(type.substr(type.length() - 1));

                    // If block was placed on top of other, replace the old one
                    bool placed_on_top = false;
                    for (int i = 0; i < this->placed_marker_blocks.size();
                         i++) {
                        if (marker_block.overwrites(
                                this->placed_marker_blocks[i])) {
                            this->placed_marker_blocks[i] = marker_block;
                            placed_on_top = true;
                            break;
                        }
                    }
                    if (!placed_on_top)
                        this->placed_marker_blocks.push_back(marker_block);

                    this->placed_marker_per_player[player_number] =
                        Tensor3(marker_block.number);
                }
                else if (json_message["header"]["message_type"] == "event" &&
                         json_message["msg"]["sub_type"] ==
                             "Event:dialogue_event") {

                    for (const auto& json_extraction :
                         json_message["data"]["extractions"]) {
                        if (json_extraction["label"] == "Agreement") {
                            this->agreement_speech_per_player[player_number] =
                                Tensor3(AGREEMENT_SPEECH);
                        }
                        else if (json_extraction["label"] == "Disagreement") {
                            this->agreement_speech_per_player[player_number] =
                                Tensor3(DISAGREEMENT_SPEECH);
                        }
                        else if (json_extraction["label"] == "Enter") {
                            this->action_enter_speech_per_player
                                [player_number] = Tensor3(ENTER_SPEECH);
                        }
                        else if (json_extraction["label"] == "MarkerMeaning") {
                            string victim_type;
                            int marker_number = NO_OBS;
                            int marker_legend = NO_OBS;

                            for (const string& attachment :
                                 json_extraction["attachments"]) {
                                if (attachment == "{\"value\":\"none\"}") {
                                    victim_type = "none";
                                }
                                else if (attachment ==
                                         "{\"value\":\"regular\"}") {
                                    victim_type = "regular";
                                }
                                else if (attachment == "{\"value\":\"1\"}") {
                                    marker_number = 1;
                                }
                                else if (attachment == "{\"value\":\"2\"}") {
                                    marker_number = 2;
                                }

                                if (victim_type == "none") {
                                    if (marker_number == 1) {
                                        marker_legend = MARKER_LEGEND_A_SPEECH;
                                    }
                                    else if (marker_number == 2) {
                                        marker_legend = MARKER_LEGEND_B_SPEECH;
                                    }
                                }
                                else if (victim_type == "regular") {
                                    if (marker_number == 1) {
                                        marker_legend = MARKER_LEGEND_B_SPEECH;
                                    }
                                    else if (marker_number == 2) {
                                        marker_legend = MARKER_LEGEND_A_SPEECH;
                                    }
                                }

                                if (marker_legend != NO_OBS) {
                                    this->marker_legend_speech_per_player
                                        [player_number] =
                                        Tensor3(marker_legend);

                                    break;
                                }
                            }
                        }
                    }
                }
                else if (json_message["header"]["message_type"] ==
                             "observation" &&
                         json_message["msg"]["sub_type"] == "FoV") {

                    for (const auto& json_block :
                         json_message["data"]["blocks"]) {
                        const string& block_type = json_block["type"];
                        if (block_type.find("victim") != string::npos) {
                            stringstream id;
                            int x = json_block["location"][0];
                            int z = json_block["location"][2];
                            id << x << "#" << z;

                            if (EXISTS(id.str(), this->rescued_victims)) {
                                this->victim_in_fov_per_player[player_number] =
                                    Tensor3(RESCUED_VICTIM_IN_FOV);
                            }
                            else {
                                if (block_type == "block_victim_1") {
                                    this->victim_in_fov_per_player
                                        [player_number] =
                                        Tensor3(REGULAR_VICTIM_IN_FOV);
                                }
                                else {
                                    this->victim_in_fov_per_player
                                        [player_number] =
                                        Tensor3(CRITICAL_VICTIM_IN_FOV);
                                }
                            }
                            break;
                        }
                    }
                }
            }
        }

        int ASISTMultiPlayerMessageConverter::get_building_section(
            int player_number) const {
            const auto& position = this->player_position.at(player_number);

            for (int section_num = 0;
                 section_num < this->building_sections.size();
                 section_num++) {

                const auto& section = this->building_sections.at(section_num);
                if (position.is_inside(section)) {
                    return section_num + 1;
                }
            }

            return OUT_OF_BUILDING;
        }

        int ASISTMultiPlayerMessageConverter::get_expanded_building_section(
            int player_number) const {
            const auto& position = this->player_position.at(player_number);

            for (int section_num = 0;
                 section_num < this->expanded_building_sections.size();
                 section_num++) {

                const auto& section =
                    this->expanded_building_sections.at(section_num);
                if (position.is_inside(section)) {
                    return section_num + 1;
                }
            }

            return OUT_OF_BUILDING;
        }

        ASISTMultiPlayerMessageConverter::Door
        ASISTMultiPlayerMessageConverter::get_closest_door(
            const Position& position) const {

            double min_dist = INT_MAX;
            Door closest_door({0, 0});
            for (const auto& door : this->doors) {
                double dist = door.position.get_distance(position);
                if (dist < min_dist) {
                    min_dist = dist;
                    closest_door = door;
                }
            }

            return closest_door;
        }

        void ASISTMultiPlayerMessageConverter::prepare_for_new_mission() {
            this->player_name_to_number.clear();
            this->player_id_to_number.clear();
            this->players.clear();

            this->task_per_player.clear();
            this->role_per_player.clear();
            this->area_per_player.clear();
            this->section_per_player.clear();
            this->marker_legend_per_player.clear();
            this->map_info_per_player.clear();
            this->placed_marker_per_player.clear();
            this->victim_in_fov_per_player.clear();
            this->agreement_speech_per_player.clear();
            this->marker_legend_speech_per_player.clear();
            this->action_enter_speech_per_player.clear();

            this->final_score = NO_OBS;
            this->map_version_assignment = NO_OBS;
            this->marker_legend_version_assignment = NO_OBS;
            this->current_team_score = 0;

            this->player_position.clear();
            this->placed_marker_blocks.clear();

            this->nearby_markers_info.clear();
            this->next_time_step = 0;

            this->rescued_victims.clear();
        }

        bool ASISTMultiPlayerMessageConverter::is_valid_message_file(
            const boost::filesystem::directory_entry& file) const {
            const string& filename = file.path().filename().string();
            return filename.find("TrialMessages") != string::npos &&
                   filename.find("Training") == string::npos &&
                   filename.find("PlanningASR") == string::npos &&
                   filename.find("FoV") == string::npos &&
                   file.path().extension().string() == ".metadata";
        }

        void ASISTMultiPlayerMessageConverter::
            do_offline_conversion_extra_validations() const {
            if (this->mission_started &&
                this->player_id_to_number.size() < this->num_players) {
                stringstream ss;
                ss << "Only " << this->player_id_to_number.size() << " out of "
                   << this->num_players << " players joined the mission.";
                throw TomcatModelException(ss.str());
            }
        }

        void ASISTMultiPlayerMessageConverter::parse_individual_message(
            const nlohmann::json& json_message) {
            if (json_message["header"]["message_type"] == "groundtruth" &&
                json_message["msg"]["sub_type"] == "measures") {
                this->json_measures = json_message;
            }
        }

        void ASISTMultiPlayerMessageConverter::write_to_log_on_mission_finished(
            nlohmann::json& json_log) const {
            json_log["nearby_markers_per_player"] = nlohmann::json::array();

            for (int player_number = 0; player_number < this->players.size();
                 player_number++) {

                nlohmann::json nearby_markers_json;
                for (const auto& [time_step, nearby_markers] :
                     this->nearby_markers_info[player_number]) {

                    nearby_markers_json[to_string(time_step)] =
                        nlohmann::json::array();

                    for (const auto& nearby_marker : nearby_markers) {
                        nlohmann::json nearby_marker_json;
                        nearby_marker_json["location"]["x"] =
                            nearby_marker.block.position.x;
                        nearby_marker_json["location"]["z"] =
                            nearby_marker.block.position.z;
                        nearby_marker_json["number"] =
                            nearby_marker.block.number;
                        nearby_marker_json["owner_player_id"] =
                            nearby_marker.block.player_id;
                        nearby_marker_json["door_id"] = nearby_marker.door.id;

                        nearby_markers_json[to_string(time_step)].push_back(
                            nearby_marker_json);
                    }
                }
                json_log["nearby_markers_per_player"].push_back(
                    nearby_markers_json);
            }
        }

    } // namespace model
} // namespace tomcat
