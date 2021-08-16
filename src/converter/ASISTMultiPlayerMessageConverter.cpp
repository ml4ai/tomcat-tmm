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
        }

        void ASISTMultiPlayerMessageConverter::load_map_area_configuration(
            const string& map_filepath) {

            this->map_area_configuration.clear();
            this->doors.clear();
            this->door_state.clear();

            fstream map_file;
            map_file.open(map_filepath);
            if (map_file.is_open()) {
                nlohmann::json json_map = nlohmann::json::parse(map_file);

                this->fill_building_sections();

                // Hallway or Room
                for (const auto& location : json_map["locations"]) {
                    const string& area_type = location["type"];
                    const string& area_id = location["id"];
                    bool is_room = area_type.find("room") != string::npos;

                    this->map_area_configuration[area_id] = is_room;

                    if (is_room && EXISTS("bounds", location)) {
                        int x1 = location["bounds"]["coordinates"][0]["x"];
                        int z1 = location["bounds"]["coordinates"][0]["z"];
                        int x2 = location["bounds"]["coordinates"][1]["x"];
                        int z2 = location["bounds"]["coordinates"][1]["z"];
                        this->rooms.push_back({x1, x2, z1, z2});
                    }
                }

                // Doors
                for (const auto& location : json_map["connections"]) {
                    const string& area_type = location["type"];
                    const string& area_id = location["id"];

                    if (area_type == "door" || area_type == "double_door") {
                        // Include each block that comprises a door as a
                        // door object in the list because the position to a
                        // doorway (in terms of blocks) will affect how
                        // marker blocks are detected.
                        int x1 = location["bounds"]["coordinates"][0]["x"];
                        int z1 = location["bounds"]["coordinates"][0]["z"];
                        Door door({x1 + 0.5, z1 + 0.5});
                        door.id = area_id;
                        this->doors.push_back(door);

                        stringstream ss;
                        ss << x1 << "#" << z1;
                        string id = ss.str();
                        this->door_state[id] = false;

                        if (area_type == "double_door") {
                            int x2 = location["bounds"]["coordinates"][1]["x"];
                            int z2 = location["bounds"]["coordinates"][1]["z"];
                            door.position = Position(x2 - 0.5, z2 - 0.5);
                            this->doors.push_back(door);

                            if (x2 - x1 > 1) {
                                ss = stringstream();
                                ss << (x2 - 1) << "#" << z1;
                                id = ss.str();
                                this->door_state[id] = false;

                                ss = stringstream();
                                ss << (x2 - 1) << "#" << z2;
                                id = ss.str();
                                this->door_state[id] = false;
                            }
                            else if (z2 - z1 > 1) {
                                ss = stringstream();
                                ss << x1 << "#" << (z2 - 1);
                                id = ss.str();
                                this->door_state[id] = false;

                                ss = stringstream();
                                ss << x2 << "#" << (z2 - 1);
                                id = ss.str();
                                this->door_state[id] = false;
                            }
                        }
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
            topics.insert("observations/events/player/door");

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
                    this->elapsed_seconds = -1;

                    // Store observations in the initial time step
                    //                    data =
                    //                    this->build_evidence_set_from_observations();
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

                    const string& team_n_trial = json_message["data"]["name"];
                    string team_id =
                        team_n_trial.substr(0, team_n_trial.find("_"));
                    string trial_id =
                        team_n_trial.substr(team_n_trial.find("_") + 1);
                    // TNo planning or planning
                    this->planning_condition =
                        2 - stoi((string)json_message["data"]["condition"]);

                    json_mission_log["trial_id"] = trial_id;
                    json_mission_log["team_id"] = team_id;
                    json_mission_log["experiment_id"] =
                        json_message["msg"]["experiment_id"];
                    json_mission_log["trial_unique_id"] =
                        json_message["msg"]["trial_id"];
                    json_mission_log["replay_parent_id"] =
                        json_message["msg"]["replay_parent_id"];
                    json_mission_log["replay_id"] =
                        json_message["msg"]["replay_id"];

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
            this->map_info_per_player.push_back(Tensor3(NO_OBS));

            // Markers
            this->marker_legend_per_player.push_back(Tensor3(NO_OBS));
            this->nearby_markers_info.resize(this->nearby_markers_info.size() +
                                             1);
            this->placed_marker_per_player.push_back(Tensor3(NO_MARKER_PLACED));

            // FoV
            this->victim_in_fov_per_player.push_back(Tensor3(NO_VICTIM_IN_FOV));
            this->safe_victim_in_fov_per_player.push_back(
                Tensor3(NO_VICTIM_IN_FOV));
            this->regular_victim_in_fov_per_player.push_back(
                Tensor3(NO_VICTIM_IN_FOV));
            this->critical_victim_in_fov_per_player.push_back(
                Tensor3(NO_VICTIM_IN_FOV));
            this->hallway_safe_victim_in_fov_per_player.push_back(
                Tensor3(NO_VICTIM_IN_FOV));
            this->hallway_regular_victim_in_fov_per_player.push_back(
                Tensor3(NO_VICTIM_IN_FOV));
            this->hallway_critical_victim_in_fov_per_player.push_back(
                Tensor3(NO_VICTIM_IN_FOV));
            this->room_safe_victim_in_fov_per_player.push_back(
                Tensor3(NO_VICTIM_IN_FOV));
            this->room_regular_victim_in_fov_per_player.push_back(
                Tensor3(NO_VICTIM_IN_FOV));
            this->room_critical_victim_in_fov_per_player.push_back(
                Tensor3(NO_VICTIM_IN_FOV));

            this->player1_marker1_in_fov_per_player.push_back(
                Tensor3(NO_NEARBY_MARKER));
            this->player2_marker1_in_fov_per_player.push_back(
                Tensor3(NO_NEARBY_MARKER));
            this->player3_marker1_in_fov_per_player.push_back(
                Tensor3(NO_NEARBY_MARKER));
            this->player1_marker2_in_fov_per_player.push_back(
                Tensor3(NO_NEARBY_MARKER));
            this->player2_marker2_in_fov_per_player.push_back(
                Tensor3(NO_NEARBY_MARKER));
            this->player3_marker2_in_fov_per_player.push_back(
                Tensor3(NO_NEARBY_MARKER));

            this->open_door_in_fov_per_player.push_back(
                Tensor3(NO_DOOR_IN_FOV));
            this->closed_door_in_fov_per_player.push_back(
                Tensor3(NO_DOOR_IN_FOV));

            // Speech
            this->marker_legend_speech_per_player.push_back(Tensor3(NO_SPEECH));
            this->agreement_speech_per_player.push_back(Tensor3(NO_SPEECH));
            this->action_move_to_room_per_player.push_back(Tensor3(NO_SPEECH));
            this->knowledge_sharing_speech_per_player.push_back(
                Tensor3(NO_SPEECH));

            // Extras
            this->player_position.push_back({0, 0});
            this->markers_near_door_per_player.push_back({});
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
                if (!json_measures["data"].empty()) {
                    // M1: Team score
                    this->final_score =
                        json_measures["data"]["M1"]["final_score"];

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
                                    "The measures agent requires the player "
                                    "name "
                                    "but that was not provided by the trial.");
                            }
                            player_number = this->player_name_to_number
                                                [json_map["player_name"]];
                        }

                        const string& static_map = json_map["map_name"];
                        string version =
                            static_map.substr(static_map.length() - 2);
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
                                    "The measures agent requires the player "
                                    "name "
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
                            this->marker_legend_version_assignment =
                                player_number;
                        }
                    }
                }
            }
        }

        EvidenceSet ASISTMultiPlayerMessageConverter::parse_after_mission_start(
            const nlohmann::json& json_message,
            nlohmann::json& json_mission_log) {

            EvidenceSet data;
            if (json_message["header"]["message_type"] == "observation" &&
                json_message["msg"]["sub_type"] == "state") {
                const string& timer = json_message["data"]["mission_timer"];
                int elapsed_seconds = this->get_elapsed_time(timer);

                if (elapsed_seconds == 0 && this->elapsed_seconds < 0) {
                    // Store initial timestamp
                    const string& timestamp = json_message["msg"]["timestamp"];
                    json_mission_log["initial_timestamp"] = timestamp;
                    json_mission_log["initial_elapsed_milliseconds"] =
                        json_message["data"]["elapsed_milliseconds"];

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

                    data = this->build_evidence_set_from_observations();
                    this->elapsed_seconds = 0;
                }

                // Every time there's a transition, we store the last
                // observations collected.
                // Some messages might be lost and we have to replicate some
                // data
                for (int t = this->elapsed_seconds + this->time_step_size;
                     t <= elapsed_seconds;
                     t++) {
                    data.hstack(this->build_evidence_set_from_observations());

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

            if (json_message["header"]["message_type"] == "groundtruth" &&
                json_message["msg"]["sub_type"] == "Mission:VictimList") {

                this->parse_victim_list_message(json_message);
                return;
            }

            string player_id;
            int player_number = -1;

            if (EXISTS("participant_id", json_message["data"])) {
                if (json_message["data"]["participant_id"] == nullptr) {
                    if (json_message["msg"]["sub_type"] != "FoV") {
                        throw TomcatModelException("Participant ID is null.");
                    }
                    // If it's FoV, we just ignore FoV Messages.
                    return;
                }
                player_id = json_message["data"]["participant_id"];
                if (EXISTS(player_id, this->player_id_to_number)) {
                    player_number = this->player_id_to_number[player_id];
                }
            }
            else if (EXISTS("playername", json_message["data"])) {
                if (json_message["data"]["playername"] == nullptr) {
                    if (json_message["msg"]["sub_type"] != "FoV") {
                        throw TomcatModelException("Participant ID is null.");
                    }
                    // If it's FoV, we just ignore FoV Messages.
                    return;
                }
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

                    this->parse_scoreboard_message(json_message);
                }
            }
            else if (player_number >= 0) {
                // Observations that are individual for each player
                if (json_message["header"]["message_type"] == "event" &&
                    json_message["msg"]["sub_type"] == "Event:ToolUsed") {

                    this->parse_tool_usage_message(json_message, player_number);
                }
                else if (json_message["header"]["message_type"] == "event" &&
                         json_message["msg"]["sub_type"] == "Event:Triage") {

                    this->parse_triage_message(json_message, player_number);
                }
                else if (json_message["header"]["message_type"] == "event" &&
                         json_message["msg"]["sub_type"] ==
                             "Event:VictimPickedUp") {

                    this->parse_victim_pickup_message(json_message,
                                                      player_number);
                }
                else if (json_message["header"]["message_type"] == "event" &&
                         json_message["msg"]["sub_type"] ==
                             "Event:VictimPlaced") {

                    this->parse_victim_placement_message(json_message,
                                                         player_number);
                }
                else if (json_message["header"]["message_type"] == "event" &&
                         json_message["msg"]["sub_type"] ==
                             "Event:RoleSelected") {

                    this->parse_role_selection_message(json_message,
                                                       player_number);
                }
                else if (json_message["header"]["message_type"] == "event" &&
                         json_message["msg"]["sub_type"] == "Event:location") {

                    this->parse_area_message(json_message, player_number);
                }
                else if (json_message["header"]["message_type"] ==
                             "observation" &&
                         json_message["msg"]["sub_type"] == "state") {

                    this->parse_player_state_message(json_message,
                                                     player_number);
                }
                else if (json_message["header"]["message_type"] == "event" &&
                         json_message["msg"]["sub_type"] ==
                             "Event:MarkerPlaced") {

                    this->parse_marker_placement_message(json_message,
                                                         player_number);
                }
                else if (json_message["header"]["message_type"] == "event" &&
                         json_message["msg"]["sub_type"] ==
                             "Event:dialogue_event") {

                    this->parse_dialog_message(json_message, player_number);
                }
                else if (json_message["header"]["message_type"] ==
                             "observation" &&
                         json_message["msg"]["sub_type"] == "FoV") {

                    this->parse_fov_message(json_message, player_number);
                }
                else if (json_message["header"]["message_type"] == "event" &&
                         json_message["msg"]["sub_type"] == "Event:Door") {

                    this->parse_door_message(json_message, player_number);
                }
            }
        }

        void ASISTMultiPlayerMessageConverter::parse_scoreboard_message(
            const nlohmann::json& json_message) {
            this->current_team_score =
                json_message["data"]["scoreboard"]["TeamScore"];
        }

        void ASISTMultiPlayerMessageConverter::parse_tool_usage_message(
            const nlohmann::json& json_message, int player_number) {
            string block_type = json_message["data"]["target_block_type"];
            alg::to_lower(block_type);
            string tool_type = json_message["data"]["tool_type"];
            alg::to_lower(tool_type);

            if (block_type.find("gravel") != string::npos &&
                tool_type == "hammer") {
                this->task_per_player[player_number] = Tensor3(CLEARING_RUBBLE);
            }
        }

        void ASISTMultiPlayerMessageConverter::parse_triage_message(
            const nlohmann::json& json_message, int player_number) {
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
            }
        }

        void ASISTMultiPlayerMessageConverter::parse_victim_pickup_message(
            const nlohmann::json& json_message, int player_number) {

            this->task_per_player[player_number] = Tensor3(CARRYING_VICTIM);

            // Remove victim from the list of visible victims
            int x = json_message["data"]["victim_x"];
            int z = json_message["data"]["victim_z"];
            stringstream id;
            id << x << "#" << z;
            this->victim_to_area.erase(id.str());
        }

        void ASISTMultiPlayerMessageConverter::parse_victim_placement_message(
            const nlohmann::json& json_message, int player_number) {

            this->task_per_player[player_number] = Tensor3(NO_TASK);

            // Add victim to the list of visible victims
            int x = json_message["data"]["victim_x"];
            int z = json_message["data"]["victim_z"];
            Position pos(x, z);
            stringstream id;
            id << x << "#" << z;
            this->victim_to_area[id.str()] = this->is_in_room(pos);
        }

        void ASISTMultiPlayerMessageConverter::parse_role_selection_message(
            const nlohmann::json& json_message, int player_number) {
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
            else if (role == "medical_specialist" || role == "medical") {
                this->role_per_player[player_number] = Tensor3(MEDICAL);
            }
            else {
                stringstream ss;
                ss << "Invalid role (" << role << ") chosen by player "
                   << this->players[player_number].id;
                throw TomcatModelException(ss.str());
            }
        }

        void ASISTMultiPlayerMessageConverter::parse_area_message(
            const nlohmann::json& json_message, int player_number) {
            if (EXISTS("locations", json_message["data"])) {
                bool area = false;
                for (const auto& location : json_message["data"]["locations"]) {

                    const string& location_id = location["id"];
                    if (location_id == "UNKNOWN") {
                        continue;
                    }

                    if (!EXISTS(location_id, this->map_area_configuration)) {
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

                this->area_per_player[player_number] = Tensor3((int)area);
            }
        }

        void ASISTMultiPlayerMessageConverter::parse_player_state_message(
            const nlohmann::json& json_message, int player_number) {
            int x = json_message["data"]["x"];
            int z = json_message["data"]["z"];
            this->player_position[player_number] = Position(x, z);
        }

        void ASISTMultiPlayerMessageConverter::parse_marker_placement_message(
            const nlohmann::json& json_message, int player_number) {
            const string& type = json_message["data"]["type"];
            int marker_number = stoi(type.substr(type.length() - 1));
            this->placed_marker_per_player[player_number] =
                Tensor3(marker_number);

            int x = json_message["data"]["marker_x"];
            int z = json_message["data"]["marker_z"];
            MarkerBlock marker({x + 0.5, z + 0.5});
            marker.number = marker_number;

            if (marker_number != 3) {
                for (const auto& door : this->doors) {
                    if (door.position.get_distance(marker.position) <=
                        MARKER_PROXIMITY_DISTANCE) {
                        this->markers_near_door_per_player[player_number]
                            .push_back(marker);
                    }
                }
            }

            // If block was placed on top of other, replace the old one
            bool placed_on_top = false;
            for (int i = 0; i < this->players.size(); i++) {
                for (int j = 0;
                     j < this->markers_near_door_per_player[i].size();
                     j++) {
                    if (marker.overwrites(
                            this->markers_near_door_per_player[i][j])) {
                        this->markers_near_door_per_player[i][j] = marker;
                        placed_on_top = true;
                        break;
                    }
                }

                if (placed_on_top) {
                    break;
                }
            }
        }

        void ASISTMultiPlayerMessageConverter::parse_dialog_message(
            const nlohmann::json& json_message, int player_number) {
            for (const auto& json_extraction :
                 json_message["data"]["extractions"]) {
                if (json_extraction["label"] == "Agreement") {
                    this->agreement_speech_per_player[player_number] =
                        Tensor3(AGREEMENT_SPEECH);
                }
                else if (json_extraction["label"] == "Enter") {
                    this->action_move_to_room_per_player[player_number] =
                        Tensor3(ENTER_SPEECH);
                }
                else if (json_extraction["label"] == "KnowledgeSharing") {
                    this->knowledge_sharing_speech_per_player[player_number] =
                        Tensor3(KNOWLEDGE_SHARING_SPEECH);
                }
                else if (json_extraction["label"] == "MarkerMeaning") {
                    string victim_type;
                    int marker_number = NO_OBS;
                    int marker_legend = NO_OBS;

                    for (const string& attachment :
                         json_extraction["attachments"]) {

                        nlohmann::json json_attachment =
                            nlohmann::json::parse(attachment);

                        if (json_attachment["value"] == "none") {
                            victim_type = json_attachment["value"];
                        }
                        else if (json_attachment["value"] == "regular") {
                            victim_type = json_attachment["value"];
                        }
                        else if (json_attachment["value"] == "1") {
                            marker_number = 1;
                        }
                        else if (json_attachment["value"] == "2") {
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
                                [player_number] = Tensor3(marker_legend);

                            break;
                        }
                    }
                }
            }
        }

        void ASISTMultiPlayerMessageConverter::parse_fov_message(
            const nlohmann::json& json_message, int player_number) {
            for (const auto& json_block : json_message["data"]["blocks"]) {
                const string& block_type = json_block["type"];
                if (block_type.find("victim") != string::npos) {
                    int x = json_block["location"][0];
                    int z = json_block["location"][2];
                    stringstream ss;
                    ss << x << "#" << z;
                    string id = ss.str();

                    bool in_room;
                    if (!EXISTS(id, this->victim_to_area)) {
                        // This is an issue that should not happen. The position
                        // of the victim placement does not match the position
                        // of the victim in FoV. This is a workaround and it
                        // assumes the victim is in the same area as the player.
                        in_room = this->area_per_player[player_number].at(
                                      0, 0, 0) == ROOM;
                    }
                    else {
                        in_room = this->victim_to_area[id];
                    }

                    // Locations are stored so we determined if the
                    // victim is inside a room or not later. No need to
                    // compute now because the granularity of the time
                    // steps are bigger than the frequency of the
                    // messages.

                    if (block_type == "block_victim_1") {
                        this->victim_in_fov_per_player[player_number] =
                            Tensor3(REGULAR_VICTIM_IN_FOV);
                        this->regular_victim_in_fov_per_player[player_number] =
                            Tensor3(VICTIM_IN_FOV);

                        if (in_room) {
                            this->room_regular_victim_in_fov_per_player
                                [player_number] = Tensor3(VICTIM_IN_FOV);
                        }
                        else {
                            this->hallway_regular_victim_in_fov_per_player
                                [player_number] = Tensor3(VICTIM_IN_FOV);
                        }
                    }
                    else if (block_type == "block_victim_proximity") {
                        this->victim_in_fov_per_player[player_number] =
                            Tensor3(CRITICAL_VICTIM_IN_FOV);
                        this->critical_victim_in_fov_per_player[player_number] =
                            Tensor3(VICTIM_IN_FOV);

                        if (in_room) {
                            this->room_critical_victim_in_fov_per_player
                                [player_number] = Tensor3(VICTIM_IN_FOV);
                        }
                        else {
                            this->hallway_critical_victim_in_fov_per_player
                                [player_number] = Tensor3(VICTIM_IN_FOV);
                        }
                    }
                    else if (block_type == "block_victim_saved") {
                        this->victim_in_fov_per_player[player_number] =
                            Tensor3(RESCUED_VICTIM_IN_FOV);
                        this->safe_victim_in_fov_per_player[player_number] =
                            Tensor3(VICTIM_IN_FOV);

                        if (in_room) {
                            this->room_safe_victim_in_fov_per_player
                                [player_number] = Tensor3(VICTIM_IN_FOV);
                        }
                        else {
                            this->hallway_safe_victim_in_fov_per_player
                                [player_number] = Tensor3(VICTIM_IN_FOV);
                        }
                    }
                }
                else if (block_type.find("marker_block") != string::npos) {
                    const string& owner = json_block["owner"];
                    int owner_number = -1;
                    if (EXISTS(owner, this->player_id_to_number)) {
                        owner_number = this->player_id_to_number[owner];
                    }
                    else if (EXISTS(owner, this->player_name_to_number)) {
                        owner_number = this->player_name_to_number[owner];
                    }

                    if (owner_number >= 0 && owner_number != player_number) {
                        if (json_block["marker_type"] == "MarkerBlock1") {
                            if (owner_number == 0) {
                                this->player1_marker1_in_fov_per_player
                                    [player_number] = Tensor3(1);
                            }
                            else if (owner_number == 1) {
                                this->player2_marker1_in_fov_per_player
                                    [player_number] = Tensor3(1);
                            }
                            else {
                                this->player3_marker1_in_fov_per_player
                                    [player_number] = Tensor3(1);
                            }
                        }
                        else if (json_block["marker_type"] == "MarkerBlock2") {
                            if (owner_number == 0) {
                                this->player1_marker2_in_fov_per_player
                                    [player_number] = Tensor3(1);
                            }
                            else if (owner_number == 1) {
                                this->player2_marker2_in_fov_per_player
                                    [player_number] = Tensor3(1);
                            }
                            else {
                                this->player3_marker2_in_fov_per_player
                                    [player_number] = Tensor3(1);
                            }
                        }
                    }
                }
                else if (block_type.find("door") != string::npos) {
                    int x = json_block["location"][0];
                    int z = json_block["location"][2];
                    stringstream ss;
                    ss << x << "#" << z;
                    string id = ss.str();

                    bool is_open = this->door_state[id];
                    if (is_open) {
                        this->open_door_in_fov_per_player[player_number] =
                            Tensor3(DOOR_IN_FOV);
                    }
                    else {
                        this->closed_door_in_fov_per_player[player_number] =
                            Tensor3(DOOR_IN_FOV);
                    }
                }
            }
        }

        void ASISTMultiPlayerMessageConverter::parse_door_message(
            const nlohmann::json& json_message, int player_number) {

            int x = json_message["data"]["door_x"];
            int z = json_message["data"]["door_z"];
            bool state = json_message["data"]["open"];

            stringstream ss;
            ss << x << "#" << z;
            string id = ss.str();

            this->door_state[id] = state;
        }

        void ASISTMultiPlayerMessageConverter::parse_victim_list_message(
            const nlohmann::json& json_message) {

            // Store the victim location as its id and whether it's in a
            // room or hallway
            for (const auto& json_victim :
                 json_message["data"]["mission_victim_list"]) {
                int x = json_victim["x"];
                int z = json_victim["z"];
                Position pos(x, z);
                stringstream id;
                id << x << "#" << z;
                this->victim_to_area[id.str()] = this->is_in_room(pos);
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

                // Marker
                data.add_data(
                    get_player_variable_label(
                        PLAYER_MARKER_LEGEND_VERSION_LABEL, player_number + 1),
                    this->marker_legend_per_player.at(player_number));
                data.add_data(
                    MARKER_LEGEND_ASSIGNMENT_LABEL,
                    Tensor3::constant(
                        1, 1, 1, this->marker_legend_version_assignment));
                data.add_data(
                    get_player_variable_label(PLAYER_PLACED_MARKER_LABEL,
                                              player_number + 1),
                    this->placed_marker_per_player.at(player_number));

                // The player detects a block if it's close enough to a block
                // and this block is close enough to a door. We only emmit a
                // marker if the player is in the hallway and the block is
                // either 1 or 2.
                int current_area =
                    this->area_per_player.at(player_number).at(0, 0, 0);
                int nearby_marker = NO_NEARBY_MARKER;
                vector<int> nearby_marker_per_player(3, nearby_marker);
                for (int i = 0; i < this->players.size(); i++) {
                    if (i == player_number)
                        continue;

                    if (current_area == ROOM)
                        continue;

                    for (int j = 0;
                         j < this->markers_near_door_per_player[i].size();
                         j++) {

                        if (this->player_position[player_number].get_distance(
                                this->markers_near_door_per_player[i][j]
                                    .position) <= MARKER_PROXIMITY_DISTANCE) {
                            nearby_marker_per_player[i] =
                                this->markers_near_door_per_player[i][j].number;

                            if (nearby_marker_per_player[i] !=
                                    NO_NEARBY_MARKER &&
                                nearby_marker_per_player[i] !=
                                    this->markers_near_door_per_player[i][j]
                                        .number) {
                                // Ignore ambiguous markers. Different markers
                                // placed by the same player close to each
                                // other.
                                nearby_marker_per_player[i] = NO_NEARBY_MARKER;
                            }
                        }
                    }
                }

                if (player_number == 0) {
                    data.add_data(
                        get_player_variable_label(PLAYER2_NEARBY_MARKER_LABEL,
                                                  player_number + 1),
                        nearby_marker_per_player[1]);
                    data.add_data(
                        get_player_variable_label(PLAYER3_NEARBY_MARKER_LABEL,
                                                  player_number + 1),
                        nearby_marker_per_player[2]);
                }
                else if (player_number == 1) {
                    data.add_data(
                        get_player_variable_label(PLAYER1_NEARBY_MARKER_LABEL,
                                                  player_number + 1),
                        nearby_marker_per_player[0]);
                    data.add_data(
                        get_player_variable_label(PLAYER3_NEARBY_MARKER_LABEL,
                                                  player_number + 1),
                        nearby_marker_per_player[2]);
                }
                else {
                    data.add_data(
                        get_player_variable_label(PLAYER1_NEARBY_MARKER_LABEL,
                                                  player_number + 1),
                        nearby_marker_per_player[0]);
                    data.add_data(
                        get_player_variable_label(PLAYER2_NEARBY_MARKER_LABEL,
                                                  player_number + 1),
                        nearby_marker_per_player[1]);
                }

                // FoV
                data.add_data(
                    get_player_variable_label(PLAYER_VICTIM_IN_FOV_LABEL,
                                              player_number + 1),
                    this->victim_in_fov_per_player[player_number]);
                data.add_data(
                    get_player_variable_label(PLAYER_SAFE_VICTIM_IN_FOV_LABEL,
                                              player_number + 1),
                    this->safe_victim_in_fov_per_player[player_number]);
                data.add_data(
                    get_player_variable_label(
                        PLAYER_REGULAR_VICTIM_IN_FOV_LABEL, player_number + 1),
                    this->regular_victim_in_fov_per_player[player_number]);
                data.add_data(
                    get_player_variable_label(
                        PLAYER_CRITICAL_VICTIM_IN_FOV_LABEL, player_number + 1),
                    this->critical_victim_in_fov_per_player[player_number]);
                data.add_data(
                    get_player_variable_label(
                        PLAYER_ROOM_SAFE_VICTIM_IN_FOV_LABEL,
                        player_number + 1),
                    this->room_safe_victim_in_fov_per_player[player_number]);
                data.add_data(
                    get_player_variable_label(
                        PLAYER_ROOM_REGULAR_VICTIM_IN_FOV_LABEL,
                        player_number + 1),
                    this->room_regular_victim_in_fov_per_player[player_number]);
                data.add_data(get_player_variable_label(
                                  PLAYER_ROOM_CRITICAL_VICTIM_IN_FOV_LABEL,
                                  player_number + 1),
                              this->room_critical_victim_in_fov_per_player
                                  [player_number]);

                // Only get markers in fov when the player is in a hallway
                if (current_area == ROOM) {
                    this->player1_marker1_in_fov_per_player[player_number] =
                        Tensor3(NO_NEARBY_MARKER);
                    this->player2_marker1_in_fov_per_player[player_number] =
                        Tensor3(NO_NEARBY_MARKER);
                    this->player3_marker1_in_fov_per_player[player_number] =
                        Tensor3(NO_NEARBY_MARKER);
                }
                if (player_number == 0) {
                    data.add_data(
                        get_player_variable_label(
                            PLAYER2_PLAYER_MARKER1_IN_FOV_LABEL,
                            player_number + 1),
                        this->player2_marker1_in_fov_per_player[player_number]);
                    data.add_data(
                        get_player_variable_label(
                            PLAYER3_PLAYER_MARKER1_IN_FOV_LABEL,
                            player_number + 1),
                        this->player3_marker1_in_fov_per_player[player_number]);
                    data.add_data(
                        get_player_variable_label(
                            PLAYER2_PLAYER_MARKER2_IN_FOV_LABEL,
                            player_number + 1),
                        this->player2_marker2_in_fov_per_player[player_number]);
                    data.add_data(
                        get_player_variable_label(
                            PLAYER3_PLAYER_MARKER2_IN_FOV_LABEL,
                            player_number + 1),
                        this->player3_marker2_in_fov_per_player[player_number]);
                }
                else if (player_number == 1) {
                    data.add_data(
                        get_player_variable_label(
                            PLAYER1_PLAYER_MARKER1_IN_FOV_LABEL,
                            player_number + 1),
                        this->player1_marker1_in_fov_per_player[player_number]);
                    data.add_data(
                        get_player_variable_label(
                            PLAYER3_PLAYER_MARKER1_IN_FOV_LABEL,
                            player_number + 1),
                        this->player3_marker1_in_fov_per_player[player_number]);
                    data.add_data(
                        get_player_variable_label(
                            PLAYER1_PLAYER_MARKER2_IN_FOV_LABEL,
                            player_number + 1),
                        this->player1_marker2_in_fov_per_player[player_number]);
                    data.add_data(
                        get_player_variable_label(
                            PLAYER3_PLAYER_MARKER2_IN_FOV_LABEL,
                            player_number + 1),
                        this->player3_marker2_in_fov_per_player[player_number]);
                }
                else {
                    data.add_data(
                        get_player_variable_label(
                            PLAYER1_PLAYER_MARKER1_IN_FOV_LABEL,
                            player_number + 1),
                        this->player1_marker1_in_fov_per_player[player_number]);
                    data.add_data(
                        get_player_variable_label(
                            PLAYER2_PLAYER_MARKER1_IN_FOV_LABEL,
                            player_number + 1),
                        this->player2_marker1_in_fov_per_player[player_number]);
                    data.add_data(
                        get_player_variable_label(
                            PLAYER1_PLAYER_MARKER2_IN_FOV_LABEL,
                            player_number + 1),
                        this->player1_marker2_in_fov_per_player[player_number]);
                    data.add_data(
                        get_player_variable_label(
                            PLAYER2_PLAYER_MARKER2_IN_FOV_LABEL,
                            player_number + 1),
                        this->player2_marker2_in_fov_per_player[player_number]);
                }

                data.add_data(
                    get_player_variable_label(
                        PLAYER_HALLWAY_SAFE_VICTIM_IN_FOV_LABEL,
                        player_number + 1),
                    this->hallway_safe_victim_in_fov_per_player[player_number]);
                data.add_data(get_player_variable_label(
                                  PLAYER_HALLWAY_REGULAR_VICTIM_IN_FOV_LABEL,
                                  player_number + 1),
                              this->hallway_regular_victim_in_fov_per_player
                                  [player_number]);
                data.add_data(get_player_variable_label(
                                  PLAYER_HALLWAY_CRITICAL_VICTIM_IN_FOV_LABEL,
                                  player_number + 1),
                              this->hallway_critical_victim_in_fov_per_player
                                  [player_number]);


                // Only get door status in fov when the player is in a hallway
                if (current_area == ROOM) {
                    this->open_door_in_fov_per_player[player_number] =
                        Tensor3(NO_DOOR_IN_FOV);
                    this->closed_door_in_fov_per_player[player_number] =
                        Tensor3(NO_DOOR_IN_FOV);
                }
                data.add_data(get_player_variable_label(OPEN_DOOR_IN_FOV_LABEL,
                                                        player_number + 1),
                              this->open_door_in_fov_per_player[player_number]);
                data.add_data(
                    get_player_variable_label(CLOSED_DOOR_IN_FOV_LABEL,
                                              player_number + 1),
                    this->closed_door_in_fov_per_player[player_number]);

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
                    get_player_variable_label(
                        PLAYER_ACTION_MOVE_TO_ROOM_SPEECH_LABEL,
                        player_number + 1),
                    this->action_move_to_room_per_player[player_number]);
                data.add_data(
                    get_player_variable_label(
                        PLAYER_KNOWLEDGE_SHARING_SPEECH_LABEL,
                        player_number + 1),
                    this->knowledge_sharing_speech_per_player[player_number]);

                // Observations from measures
                if (!json_measures.empty()) {
                    // Final score
                    data.add_data(
                        FINAL_TEAM_SCORE_LABEL,
                        Tensor3::constant(1, 1, 1, this->final_score));
                }

                // Team
                data.add_data(
                    TEAM_SCORE_LABEL,
                    Tensor3::constant(1, 1, 1, this->current_team_score));

                // Reset observations

                // Carrying ans saving a victim are tasks that have
                // an explicit end event. We don't reset the last
                // observation for this task then. It's going to be
                // reset when its ending is detected.
                int last_task =
                    this->task_per_player.at(player_number)(0, 0)(0, 0);
                if (last_task == CLEARING_RUBBLE) {
                    this->task_per_player[player_number] = Tensor3(NO_TASK);
                }

                // Marker
                this->placed_marker_per_player[player_number] =
                    Tensor3(NO_MARKER_PLACED);

                // FoV
                this->victim_in_fov_per_player[player_number] =
                    Tensor3(NO_VICTIM_IN_FOV);
                this->safe_victim_in_fov_per_player[player_number] =
                    Tensor3(NO_VICTIM_IN_FOV);
                this->regular_victim_in_fov_per_player[player_number] =
                    Tensor3(NO_VICTIM_IN_FOV);
                this->critical_victim_in_fov_per_player[player_number] =
                    Tensor3(NO_VICTIM_IN_FOV);
                this->hallway_safe_victim_in_fov_per_player[player_number] =
                    Tensor3(NO_VICTIM_IN_FOV);
                this->hallway_regular_victim_in_fov_per_player[player_number] =
                    Tensor3(NO_VICTIM_IN_FOV);
                this->hallway_critical_victim_in_fov_per_player[player_number] =
                    Tensor3(NO_VICTIM_IN_FOV);
                this->room_safe_victim_in_fov_per_player[player_number] =
                    Tensor3(NO_VICTIM_IN_FOV);
                this->room_regular_victim_in_fov_per_player[player_number] =
                    Tensor3(NO_VICTIM_IN_FOV);
                this->room_critical_victim_in_fov_per_player[player_number] =
                    Tensor3(NO_VICTIM_IN_FOV);

                this->player1_marker1_in_fov_per_player[player_number] =
                    Tensor3(NO_NEARBY_MARKER);
                this->player2_marker1_in_fov_per_player[player_number] =
                    Tensor3(NO_NEARBY_MARKER);
                this->player3_marker1_in_fov_per_player[player_number] =
                    Tensor3(NO_NEARBY_MARKER);
                this->player1_marker2_in_fov_per_player[player_number] =
                    Tensor3(NO_NEARBY_MARKER);
                this->player2_marker2_in_fov_per_player[player_number] =
                    Tensor3(NO_NEARBY_MARKER);
                this->player3_marker2_in_fov_per_player[player_number] =
                    Tensor3(NO_NEARBY_MARKER);

                this->open_door_in_fov_per_player[player_number] =
                    Tensor3(NO_DOOR_IN_FOV);
                this->closed_door_in_fov_per_player[player_number] =
                    Tensor3(NO_DOOR_IN_FOV);

                // Reset speeches
                this->agreement_speech_per_player[player_number] =
                    Tensor3(NO_SPEECH);
                this->marker_legend_speech_per_player[player_number] =
                    Tensor3(NO_SPEECH);
                this->action_move_to_room_per_player[player_number] =
                    Tensor3(NO_SPEECH);
                this->knowledge_sharing_speech_per_player[player_number] =
                    Tensor3(NO_SPEECH);
            }

            this->next_time_step += 1;

            return data;
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
            this->map_info_per_player.clear();
            this->victim_to_area.clear();

            // FoV
            this->victim_in_fov_per_player.clear();
            this->safe_victim_in_fov_per_player.clear();
            this->regular_victim_in_fov_per_player.clear();
            this->critical_victim_in_fov_per_player.clear();
            this->hallway_safe_victim_in_fov_per_player.clear();
            this->hallway_regular_victim_in_fov_per_player.clear();
            this->hallway_critical_victim_in_fov_per_player.clear();
            this->room_safe_victim_in_fov_per_player.clear();
            this->room_regular_victim_in_fov_per_player.clear();
            this->room_critical_victim_in_fov_per_player.clear();

            this->player1_marker1_in_fov_per_player.clear();
            this->player2_marker1_in_fov_per_player.clear();
            this->player3_marker1_in_fov_per_player.clear();
            this->player1_marker2_in_fov_per_player.clear();
            this->player2_marker2_in_fov_per_player.clear();
            this->player3_marker2_in_fov_per_player.clear();

            this->open_door_in_fov_per_player.clear();
            this->closed_door_in_fov_per_player.clear();

            // Marker
            this->marker_legend_per_player.clear();
            this->placed_marker_per_player.clear();

            // Speech
            this->agreement_speech_per_player.clear();
            this->marker_legend_speech_per_player.clear();
            this->action_move_to_room_per_player.clear();
            this->knowledge_sharing_speech_per_player.clear();

            // Team
            this->planning_condition = NO_OBS;
            this->final_score = NO_OBS;
            this->map_version_assignment = NO_OBS;
            this->marker_legend_version_assignment = NO_OBS;
            this->current_team_score = 0;

            // Extras
            this->player_position.clear();
            this->placed_marker_blocks.clear();
            this->markers_near_door_per_player.clear();

            this->nearby_markers_info.clear();
            this->next_time_step = 0;
        }

        bool ASISTMultiPlayerMessageConverter::is_valid_message_file(
            const boost::filesystem::directory_entry& file) const {
            const string filename = file.path().filename().string();

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
            // No longer needed. This will be given in an external file.

            //            json_log["nearby_markers_per_player"] =
            //            nlohmann::json::array();
            //
            //            for (int player_number = 0; player_number <
            //            this->players.size();
            //                 player_number++) {
            //
            //                nlohmann::json nearby_markers_json;
            //                for (const auto& [time_step, nearby_markers] :
            //                     this->nearby_markers_info[player_number]) {
            //
            //                    nearby_markers_json[to_string(time_step)] =
            //                        nlohmann::json::array();
            //
            //                    for (const auto& nearby_marker :
            //                    nearby_markers) {
            //                        nlohmann::json nearby_marker_json;
            //                        nearby_marker_json["location"]["x"] =
            //                            nearby_marker.block.position.x;
            //                        nearby_marker_json["location"]["z"] =
            //                            nearby_marker.block.position.z;
            //                        nearby_marker_json["number"] =
            //                            nearby_marker.block.number;
            //                        nearby_marker_json["owner_player_id"] =
            //                            this->players[nearby_marker.block.player_number].id;
            //                        nearby_marker_json["door_id"] =
            //                        nearby_marker.door.id;
            //
            //                        nearby_markers_json[to_string(time_step)].push_back(
            //                            nearby_marker_json);
            //                    }
            //                }
            //                json_log["nearby_markers_per_player"].push_back(
            //                    nearby_markers_json);
            //            }
        }

        bool ASISTMultiPlayerMessageConverter::is_in_room(
            const Position& position) const {
            for (const auto& room : this->rooms) {
                if (position.is_inside(room)) {
                    return true;
                }
            }

            return false;
        }

    } // namespace model
} // namespace tomcat
