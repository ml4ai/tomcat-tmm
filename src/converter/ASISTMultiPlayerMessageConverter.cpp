#include "ASISTMultiPlayerMessageConverter.h"

#include <fstream>
#include <iomanip>

#include <boost/algorithm/string.hpp>
#include <fmt/format.h>

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
        // Member functions
        //----------------------------------------------------------------------
        void ASISTMultiPlayerMessageConverter::copy_converter(
            const ASISTMultiPlayerMessageConverter& converter) {
            ASISTMessageConverter::copy_converter(converter);
            this->mission_started = converter.mission_started;
            this->elapsed_time = converter.elapsed_time;
            this->num_players = converter.num_players;
            this->player_name_to_id = converter.player_name_to_id;
            this->task_per_player = converter.task_per_player;
            this->role_per_player = converter.role_per_player;
        }

        void ASISTMultiPlayerMessageConverter::load_map_area_configuration(
            const string& map_filepath) {

            fstream map_file;
            map_file.open(map_filepath);
            if (map_file.is_open()) {
                nlohmann::json json_map = nlohmann::json::parse(map_file);

                // Bounding box of the map
                int min_x = INT32_MAX;
                int max_x = INT32_MIN;
                int min_z = INT32_MAX;
                int max_z = INT32_MIN;

                for (const auto& location : json_map["locations"]) {
                    const string& area_type = location["type"];
                    const string& area_id = location["id"];

                    if (area_id.find("sga") == string::npos) {
                        this->map_area_configuration[area_id] =
                            area_type.find("room") != string::npos;
                    }
                    else {
                        // Staging area is reported as a room. Here I
                        // overwrite because I want to report as a room, only
                        // areas where there can be a victim in it.
                        this->map_area_configuration[area_id] = false;
                    }

                    if (EXISTS("bounds", location)) {
                        for (const auto& coordinate :
                             location["bounds"]["coordinates"]) {
                            if (coordinate["x"] < min_x) {
                                min_x = coordinate["x"];
                            }
                            if (coordinate["x"] > max_x) {
                                max_x = coordinate["x"];
                            }
                            if (coordinate["z"] < min_z) {
                                min_z = coordinate["z"];
                            }
                            if (coordinate["z"] > max_x) {
                                max_z = coordinate["z"];
                            }
                        }
                    }
                }

                // Split the map into 6 non-overlapping sections
                int section_width = (max_x - min_x) / 3;
                int section_height = (max_z - min_z) / 2;

                BoundingBox section1(min_x,
                                     min_x + section_width,
                                     min_z,
                                     min_z + section_height);
                BoundingBox section2(min_x,
                                     min_x + section_width,
                                     min_z + section_height,
                                     max_z);
                BoundingBox section3(min_x + section_width,
                                     min_x + 2 * section_width,
                                     min_z,
                                     min_z + section_height);
                BoundingBox section4(min_x + section_width,
                                     min_x + 2 * section_width,
                                     min_z + section_height,
                                     max_z);
                BoundingBox section5(min_x + 2 * section_width,
                                     max_x,
                                     min_z,
                                     min_z + section_height);
                BoundingBox section6(min_x + 2 * section_width,
                                     max_x,
                                     min_z + section_height,
                                     max_z);

                this->building_sections.push_back(section1);
                this->building_sections.push_back(section2);
                this->building_sections.push_back(section3);
                this->building_sections.push_back(section4);
                this->building_sections.push_back(section5);
                this->building_sections.push_back(section6);
            }
            else {
                stringstream ss;
                ss << "Map configuration file in " << map_filepath
                   << " does not exist.";
                throw TomcatModelException(ss.str());
            }
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

            return topics;
        }

        EvidenceSet
        ASISTMultiPlayerMessageConverter::parse_before_mission_start(
            const nlohmann::json& json_message,
            nlohmann::json& json_mission_log) {

            EvidenceSet data;
            if (json_message["header"]["message_type"] == "observation" &&
                json_message["msg"]["sub_type"] == "state") {
                const string& player_name = json_message["data"]["playername"];

                // Detect new players and add to the list
                if (player_name != "ASU_MC") {
                    if (!EXISTS(player_name, this->player_name_to_id) &&
                        this->player_name_to_id.size() < this->num_players) {
                        this->add_player(player_name);
                    }
                }
            }
            else if (json_message["header"]["message_type"] == "event" &&
                     json_message["msg"]["sub_type"] == "Event:MissionState") {
                string mission_state = json_message["data"]["mission_state"];
                alg::to_lower(mission_state);

                if (mission_state == "start") {
                    this->mission_started = true;
                    this->elapsed_time = this->time_step_size;

                    // Add list of players to the log
                    if (!json_mission_log.contains("players")) {
                        json_mission_log["players"] = nlohmann::json::array();
                        for (int i = 0; i < this->player_name_to_id.size();
                             i++) {
                            json_mission_log["players"].push_back("");
                        }
                    }

                    for (const auto& [player_name, player_id] :
                         this->player_name_to_id) {
                        json_mission_log["players"][player_id] = player_name;
                    }

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
                    json_mission_log["trial"] =
                        json_message["data"]["trial_number"];
                    json_mission_log["experiment_id"] =
                        json_message["msg"]["experiment_id"];

                    if (EXISTS("client_info", json_message["data"])) {
                        this->fill_client_info_data(
                            json_message["data"]["client_info"]);
                    }
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

        void ASISTMultiPlayerMessageConverter::add_player(
            const string& player_name) {

            int id = this->player_name_to_id.size();
            this->player_name_to_id[player_name] = id;

            this->task_per_player.push_back(Tensor3(NO_TASK));
            this->role_per_player.push_back(Tensor3(NO_ROLE));
            this->area_per_player.push_back(Tensor3(HALLWAY));
            this->section_per_player.push_back(Tensor3(OUT_OF_BUILDING));
            this->marker_legend_per_player.push_back(Tensor3(NO_OBS));
            this->map_info_per_player.push_back(Tensor3(NO_OBS));
            this->seen_marker_per_player.push_back(Tensor3(NO_OBS));

            this->player_position.push_back({0, 0});
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

        void ASISTMultiPlayerMessageConverter::fill_client_info_data(
            const nlohmann::json& json_client_info) {

            for (const auto& info_per_player : json_client_info) {
                const string& player_name = info_per_player["playername"];
                if (!EXISTS(player_name, this->player_name_to_id) &&
                    this->player_name_to_id.size() < this->num_players) {
                    this->add_player(player_name);
                }

                int player_id = this->player_name_to_id.at(player_name);

                // Map info
                const string& map_version = info_per_player["staticmapversion"];
                if (map_version.find("24") != string::npos) {
                    this->map_info_per_player[player_id] =
                        Tensor3(SECTIONS_2N4);
                }
                else if (map_version.find("34") != string::npos) {
                    this->map_info_per_player[player_id] =
                        Tensor3(SECTIONS_3N4);
                }
                else if (map_version.find("64") != string::npos) {
                    this->map_info_per_player[player_id] =
                        Tensor3(SECTIONS_6N4);
                }

                // Marker legend
                const string& marker_legend =
                    info_per_player["markerblocklegend"];
                if (marker_legend.at(0) == 'A') {
                    this->marker_legend_per_player[player_id] =
                        Tensor3(MARKER_LEGEND_A);
                }
                else if (marker_legend.at(0) == 'B') {
                    this->marker_legend_per_player[player_id] =
                        Tensor3(MARKER_LEGEND_B);
                }
            }
        }

        EvidenceSet ASISTMultiPlayerMessageConverter::
            build_evidence_set_from_observations() {

            EvidenceSet data;
            for (int player_id = 0; player_id < this->player_name_to_id.size();
                 player_id++) {
                data.add_data(fmt::format("{}#P{}", TASK, player_id + 1),
                              this->task_per_player.at(player_id));
                data.add_data(fmt::format("{}#P{}", ROLE, player_id + 1),
                              this->role_per_player.at(player_id));
                data.add_data(fmt::format("{}#P{}", AREA, player_id + 1),
                              this->area_per_player.at(player_id));
                data.add_data(fmt::format("{}#P{}", SEEN_MARKER, player_id + 1),
                              this->seen_marker_per_player.at(player_id));

                int section = this->get_building_section(player_id);
                data.add_data(fmt::format("{}#P{}", SECTION, player_id + 1),
                              section);

                data.add_data(fmt::format("{}#P{}", MAP_INFO, player_id + 1),
                              this->map_info_per_player.at(player_id));
                data.add_data(
                    fmt::format("{}#P{}", MARKER_LEGEND, player_id + 1),
                    this->marker_legend_per_player.at(player_id));

                // Carrying ans saving a victim are tasks that have
                // an explicit end event. We don't reset the last
                // observation for this task then. It's going to be
                // reset when its ending is detected.
                int last_task = this->task_per_player.at(player_id)(0, 0)(0, 0);
                if (last_task == CLEARING_RUBBLE) {
                    this->task_per_player[player_id] = Tensor3(NO_TASK);
                }
            }

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
                        this->mission_finished = true;
                    }
                }
            }
            else if (json_message["header"]["message_type"] == "trial" &&
                     json_message["msg"]["sub_type"] == "stop") {
                this->mission_finished = true;
            }

            if (!this->mission_finished) {
                this->fill_observation(json_message);
            }

            return data;
        }

        void ASISTMultiPlayerMessageConverter::fill_observation(
            const nlohmann::json& json_message) {

            if (!EXISTS("playername", json_message["data"])) {
                return;
            }

            const string& player_name = json_message["data"]["playername"];

            if (!EXISTS(player_name, this->player_name_to_id)) {
                return;
            }

            int player_id = this->player_name_to_id.at(player_name);

            if (json_message["header"]["message_type"] == "event" &&
                json_message["msg"]["sub_type"] == "event:ToolUsed") {

                string block_type = json_message["data"]["target_block_type"];
                alg::to_lower(block_type);
                string tool_type = json_message["data"]["tool_type"];
                alg::to_lower(tool_type);

                if (block_type.find("gravel") != string::npos &&
                    tool_type == "hammer") {
                    this->task_per_player[player_id] = Tensor3(CLEARING_RUBBLE);
                }
            }
            else if (json_message["header"]["message_type"] == "event" &&
                     json_message["msg"]["sub_type"] == "Event:Triage") {
                if (json_message["data"]["triage_state"] == "IN_PROGRESS") {
                    if (EXISTS("type", json_message["data"])) {
                        string victim_type = json_message["data"]["type"];
                        alg::to_lower(victim_type);

                        if (victim_type == "regular") {
                            this->task_per_player[player_id] =
                                Tensor3(SAVING_REGULAR);
                        }
                        else if (victim_type == "critical") {
                            this->task_per_player[player_id] =
                                Tensor3(SAVING_CRITICAL);
                        }
                    }
                    else {
                        // Old format
                        string victim_color = json_message["data"]["color"];
                        alg::to_lower(victim_color);

                        if (victim_color == "green") {
                            this->task_per_player[player_id] =
                                Tensor3(SAVING_REGULAR);
                        }
                        else if (victim_color == "yellow") {
                            this->task_per_player[player_id] =
                                Tensor3(SAVING_CRITICAL);
                        }
                    }
                }
                else {
                    this->task_per_player[player_id] = Tensor3(NO_TASK);
                }
            }
            else if (json_message["header"]["message_type"] == "event" &&
                     json_message["msg"]["sub_type"] ==
                         "Event:VictimPickedUp") {
                this->task_per_player[player_id] = Tensor3(CARRYING_VICTIM);
            }
            else if (json_message["header"]["message_type"] == "event" &&
                     json_message["msg"]["sub_type"] == "Event:VictimPlaced") {
                this->task_per_player[player_id] = Tensor3(NO_TASK);
            }
            else if (json_message["header"]["message_type"] == "event" &&
                     json_message["msg"]["sub_type"] == "Event:RoleSelected") {
                string role = json_message["data"]["new_role"];
                alg::to_lower(role);

                if (role == "search_specialist" || role == "search") {
                    this->role_per_player[player_id] = Tensor3(SEARCH);
                }
                else if (role == "hazardous_material_specialist" ||
                         role == "hammer") {
                    this->role_per_player[player_id] = Tensor3(HAMMER);
                }
                else if (role == "medical_specialist" || role == "medical") {
                    this->role_per_player[player_id] = Tensor3(MEDICAL);
                }
                else if (role == "none") {
                    this->role_per_player[player_id] = Tensor3(NO_ROLE);
                }
                else {
                    stringstream ss;
                    ss << "Invalid role (" << role << ") chosen by player "
                       << player_name;
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

                    this->area_per_player[player_id] = Tensor3((int)area);
                }
            }
            else if (json_message["header"]["message_type"] == "observation" &&
                     json_message["msg"]["sub_type"] == "state") {
                int x = json_message["data"]["x"];
                int z = json_message["data"]["z"];
                this->player_position[player_id] = Position(x, z);
            }
        }

        int ASISTMultiPlayerMessageConverter::get_building_section(
            int player_id) const {
            const auto& position = this->player_position.at(player_id);

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

        void ASISTMultiPlayerMessageConverter::prepare_for_new_mission() {
            this->player_name_to_id.clear();
            this->task_per_player.clear();
            this->role_per_player.clear();
            this->area_per_player.clear();
            this->section_per_player.clear();
            this->marker_legend_per_player.clear();
            this->map_info_per_player.clear();
            this->seen_marker_per_player.clear();
        }

        bool ASISTMultiPlayerMessageConverter::is_valid_message_file(
            const boost::filesystem::directory_entry& file) const {
            const string& filename = file.path().filename().string();
            return filename.find("TrialMessages") != string::npos &&
                   filename.find("Training") == string::npos &&
                   filename.find("PlanningASR") == string::npos &&
                   file.path().extension().string() == ".metadata";
        }

        void ASISTMultiPlayerMessageConverter::
            do_offline_conversion_extra_validations() const {
            if (this->mission_started &&
                this->player_name_to_id.size() < this->num_players) {
                stringstream ss;
                ss << "Only " << this->player_name_to_id.size() << " out of "
                   << this->num_players << " players joined the mission.";
                throw TomcatModelException(ss.str());
            }
        }

    } // namespace model
} // namespace tomcat
