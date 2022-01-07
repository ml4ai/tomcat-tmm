#include "ASISTStudy3MessageConverter.h"

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
        ASISTStudy3MessageConverter::ASISTStudy3MessageConverter(
            int num_seconds,
            int time_step_size,
            const std::string& map_filepath,
            int num_players)
            : ASISTMessageConverter(num_seconds, time_step_size),
              num_players(num_players) {

            this->load_map_area_configuration(map_filepath);
        }

        ASISTStudy3MessageConverter::~ASISTStudy3MessageConverter() {}

        //----------------------------------------------------------------------
        // Copy & Move constructors/assignments
        //----------------------------------------------------------------------
        ASISTStudy3MessageConverter::ASISTStudy3MessageConverter(
            const ASISTStudy3MessageConverter& converter)
            : ASISTMessageConverter(converter.time_steps *
                                        converter.time_step_size,
                                    converter.time_step_size) {
            this->copy_converter(converter);
        }

        ASISTStudy3MessageConverter& ASISTStudy3MessageConverter::operator=(
            const ASISTStudy3MessageConverter& converter) {
            this->copy_converter(converter);
            return *this;
        }

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        void ASISTStudy3MessageConverter::copy_converter(
            const ASISTStudy3MessageConverter& converter) {
            ASISTMessageConverter::copy_converter(converter);
        }

        void ASISTStudy3MessageConverter::load_map_area_configuration(
            const string& map_filepath) {

            fstream map_file;
            map_file.open(map_filepath);
            if (map_file.is_open()) {
                nlohmann::json json_map = nlohmann::json::parse(map_file);

                int min_x = INT_MAX;
                int max_x = INT_MIN;
                int min_z = INT_MAX;
                int max_z = INT_MIN;

                for (const auto& location : json_map["locations"]) {
                    if (EXISTS("bounds", location)) {
                        int x1 = location["bounds"]["coordinates"][0]["x"];
                        int z1 = location["bounds"]["coordinates"][0]["z"];
                        int x2 = location["bounds"]["coordinates"][1]["x"];
                        int z2 = location["bounds"]["coordinates"][1]["z"];

                        // Smallest and largest coordinates to define the
                        // bounding box of the map
                        min_x = min(min_x, x1);
                        min_x = min(min_x, x2);
                        max_x = max(min_x, x1);
                        max_x = max(min_x, x2);
                        min_z = min(min_z, z1);
                        min_z = min(min_z, z2);
                        max_z = max(min_z, z1);
                        max_z = max(min_z, z2);
                    }
                }

                BoundingBox map_bounding_box(min_x, max_x, min_z, max_z);
                this->fill_map_sections(map_bounding_box);
            }
        }

        void ASISTStudy3MessageConverter::fill_map_sections(
            const BoundingBox& map_bounding_box) {

            /**
             * 6 Map Sections
             * 1 2 3
             * 4 5 6
             */

            this->map_sections.clear();

            auto top_section = map_bounding_box.get_horizontal_splits()[0];
            auto bottom_section = map_bounding_box.get_horizontal_splits()[1];

            auto top_sections = top_section.get_vertical_splits(3);
            auto bottom_sections = bottom_section.get_vertical_splits(3);

            this->map_sections = top_sections;
            this->map_sections.insert(this->map_sections.end(),
                                      bottom_sections.begin(),
                                      bottom_sections.end());
        }

        unordered_set<string>
        ASISTStudy3MessageConverter::get_used_topics() const {
            unordered_set<string> topics;

            topics.insert("trial");
            topics.insert("observations/events/mission");
            topics.insert("observations/state");
            topics.insert("observations/events/scoreboard");
            topics.insert("observations/events/player/role_selected");

            return topics;
        }

        EvidenceSet ASISTStudy3MessageConverter::parse_before_mission_start(
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
                    string team_id = "";
                    //                        team_n_trial.substr(0,
                    //                        team_n_trial.find("_"));
                    string trial_id = team_n_trial;
                    //                        team_n_trial.substr(team_n_trial.find("_")
                    //                        + 1);

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

        void ASISTStudy3MessageConverter::add_player(const Player& player) {
            int player_number = this->player_id_to_number.size();
            this->player_id_to_number[player.id] = player_number;
            this->players.push_back(player);

            // Evidence per player
            this->player_role.push_back(MEDICAL_ROLE);
            this->player_position.push_back({});
            this->player_seconds_in_map_section.push_back(
                vector<int>(this->map_sections.size(), 0));
        }

        int ASISTStudy3MessageConverter::get_numeric_trial_number(
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

        void ASISTStudy3MessageConverter::fill_players(
            const nlohmann::json& json_client_info, nlohmann::json& json_log) {

            json_log["players"] = nlohmann::json::array();

            for (const auto& info_per_player : json_client_info) {
                Player player;
                player.id = info_per_player["participant_id"];
                player.callsign = info_per_player["callsign"];

                if (EXISTS("playername", info_per_player)) {
                    player.name = info_per_player["playername"];
                }
                this->add_player(player);

                nlohmann::json json_player;
                json_player["id"] = player.id;
                json_player["callsign"] = player.callsign;
                json_player["name"] = player.name;
                json_log["players"].push_back(json_player);
            }
        }

        EvidenceSet ASISTStudy3MessageConverter::parse_after_mission_start(
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

        void ASISTStudy3MessageConverter::fill_observation(
            const nlohmann::json& json_message) {

            if (json_message["header"]["message_type"] == "observation" &&
                json_message["msg"]["sub_type"] == "Event:Scoreboard") {
                // Team evidence
                this->parse_scoreboard_message(json_message);
            }
            else {
                string player_id;
                int player_number = -1;

                if (EXISTS("participant_id", json_message["data"])) {
                    if (json_message["data"]["participant_id"] == nullptr) {
                        if (json_message["msg"]["sub_type"] != "FoV") {
                            throw TomcatModelException(
                                "Participant ID is null.");
                        }
                        // If it's FoV, we just ignore FoV Messages.
                        return;
                    }
                    player_id = json_message["data"]["participant_id"];
                    if (EXISTS(player_id, this->player_id_to_number)) {
                        player_number = this->player_id_to_number[player_id];
                    }
                }

                if (player_id != "") {
                    // Player evidence
                    if (json_message["header"]["message_type"] == "event" &&
                        json_message["msg"]["sub_type"] ==
                            "Event:RoleSelected") {
                        this->parse_role_selection_message(json_message,
                                                           player_number);
                    }
                    else if (json_message["header"]["message_type"] ==
                                 "observation" &&
                             json_message["msg"]["sub_type"] == "state") {
                        this->parse_player_state_message(json_message,
                                                         player_number);
                    }
                }
            }
        }

        void ASISTStudy3MessageConverter::parse_scoreboard_message(
            const nlohmann::json& json_message) {
            int score = json_message["data"]["scoreboard"]["TeamScore"];
            this->team_score(0, 0, 0) = score;
        }

        void ASISTStudy3MessageConverter::parse_role_selection_message(
            const nlohmann::json& json_message, int player_number) {
            string new_role = json_message["data"]["new_role"];
            alg::to_lower(new_role);

            if (new_role == "transport_specialist") {
                this->player_role[player_number] = Tensor3(TRANSPORTER_ROLE);
            }
            else if (new_role == "engineering_specialist") {
                this->player_role[player_number] = Tensor3(ENGINEER_ROLE);
            }
            else if (new_role == "medical_specialist") {
                this->player_role[player_number] = Tensor3(MEDICAL_ROLE);
            }
            else {
                throw TomcatModelException(
                    fmt::format("Role {} invalid.", new_role));
            }
        }

        void ASISTStudy3MessageConverter::parse_player_state_message(
            const nlohmann::json& json_message, int player_number) {
            int x = json_message["data"]["x"];
            int z = json_message["data"]["z"];
            this->player_position[player_number] = Position(x, z);
        }

        EvidenceSet
        ASISTStudy3MessageConverter::build_evidence_set_from_observations() {
            EvidenceSet data;

            // Per team
            data.add_data(TEAM_SCORE, this->team_score);
            data.add_data(ELAPSED_SECONDS, Tensor3(this->next_time_step + 1));

            // Per player
            for (int player_number = 0; player_number < this->num_players;
                 player_number++) {
                this->collect_player_seconds_in_map_section(data,
                                                            player_number);

                string node_label =
                    this->get_player_variable_label(PLAYER_ROLE, player_number);
                data.add_data(node_label, this->player_role[player_number]);
            }

            this->next_time_step += 1;
            return data;
        }

        void ASISTStudy3MessageConverter::prepare_for_new_mission() {
            this->next_time_step = 0;
            this->team_score = Tensor3(0);
            this->players.clear();
            this->player_id_to_number.clear();
            this->player_role.clear();
            this->player_position.clear();
            this->player_seconds_in_map_section.clear();
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
                this->player_id_to_number.size() < this->num_players) {
                stringstream ss;
                ss << "Only " << this->player_id_to_number.size() << " out of "
                   << this->num_players << " players joined the mission.";
                throw TomcatModelException(ss.str());
            }
        }

        void ASISTStudy3MessageConverter::parse_individual_message(
            const nlohmann::json& json_message) {}

        void ASISTStudy3MessageConverter::write_to_log_on_mission_finished(
            nlohmann::json& json_log) const {}

        void ASISTStudy3MessageConverter::collect_player_seconds_in_map_section(
            EvidenceSet& data, int player_number) {
            int map_section = this->get_player_map_section(player_number);
            this->player_seconds_in_map_section[player_number][map_section] +=
                this->time_step_size;

            for (int section_number = 0;
                 section_number < this->map_sections.size();
                 section_number++) {

                string node_label = this->get_player_variable_label(
                    fmt::format("{}{}",
                                ELAPSED_SECONDS_MAP_SECTION,
                                section_number + 1),
                    player_number);
                Tensor3 elapsed_seconds(
                    this->player_seconds_in_map_section[player_number]
                                                       [section_number]);

                data.add_data(node_label, elapsed_seconds);
            }
        }

        int ASISTStudy3MessageConverter::get_player_map_section(
            int player_number) const {
            for (int section_number = 0;
                 section_number < this->map_sections.size();
                 section_number++) {

                if (this->player_position[player_number].is_inside(
                        this->map_sections[section_number])) {
                    return section_number;
                }
            }

            return 5; // Middle section where players start the game
        }
    } // namespace model
} // namespace tomcat
