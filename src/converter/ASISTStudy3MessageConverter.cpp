#include "ASISTStudy3MessageConverter.h"

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
            const string& map_filepath) {}

        unordered_set<string>
        ASISTStudy3MessageConverter::get_used_topics() const {
            unordered_set<string> topics;

            topics.insert("trial");
            topics.insert("observations/events/mission");
            topics.insert("observations/state");
            topics.insert("observations/events/scoreboard");
            topics.insert("agent/control/rollcall/request");

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

            // Clear player evidence from previous mission
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
            else if (json_message["header"]["message_type"] == "agent" &&
                     json_message["msg"]["sub_type"] == "rollcall:request") {
                if (this->callback_function) {
                    this->callback_function(json_message);
                }
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
                }
            }
        }

        void ASISTStudy3MessageConverter::parse_scoreboard_message(
            const nlohmann::json& json_message) {
            int score = json_message["data"]["scoreboard"]["TeamScore"];
            this->team_score(0, 0, 0) = score;
        }

        EvidenceSet
        ASISTStudy3MessageConverter::build_evidence_set_from_observations() {
            EvidenceSet data;

            // Per team
            data.add_data(TEAM_SCORE, this->team_score);
            data.add_data(ELAPSED_SECOND, Tensor3(this->next_time_step + 1));

            this->next_time_step += 1;
            return data;
        }

        void ASISTStudy3MessageConverter::prepare_for_new_mission() {
            this->next_time_step = 0;
            this->team_score = Tensor3(0);
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
    } // namespace model
} // namespace tomcat
