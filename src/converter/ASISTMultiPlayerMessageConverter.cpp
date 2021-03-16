#include "ASISTMultiPlayerMessageConverter.h"

#include <fstream>
#include <iomanip>

#include <fmt/format.h>

// Task values
#define NO_TASK 0
#define CARRYING_VICTIM 1
#define CLEARING_RUBBLE 2
#define SAVING_REGULAR 3
#define SAVING_CRITICAL 4

// Role values
#define NO_ROLE 0
#define SEARCH 1
#define HAMMER 2
#define MEDICAL 3

namespace tomcat {
    namespace model {

        using namespace std;

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

            // TODO when location monitor is available in the messages
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
                        int id = this->player_name_to_id.size();
                        this->player_name_to_id[player_name] = id;

                        // Players start in the moving/waiting state,
                        // with no role associated
                        this->task_per_player.push_back({Tensor3(NO_TASK)});
                        this->role_per_player.push_back({Tensor3(NO_TASK)});

                        if (!json_mission_log.contains("players")) {
                            json_mission_log["players"] =
                                nlohmann::json::array();
                        }
                        json_mission_log["players"].push_back(player_name);
                    }
                }
            }
            else if (json_message["header"]["message_type"] == "event" &&
                     json_message["msg"]["sub_type"] == "Event:MissionState") {
                if (json_message["data"]["mission_state"] == "Start") {
                    this->mission_started = true;
                    this->elapsed_time = this->time_step_size;
                    this->num_players = this->player_name_to_id.size();

                    for (int i = 0; i < this->num_players; i++) {
                        data.add_data(fmt::format("TaskP{}", i + 1),
                                      this->task_per_player.at(i));
                        data.add_data(fmt::format("RoleP{}", i + 1),
                                      this->role_per_player.at(i));

                        // Carrying ans saving a victim are tasks that have
                        // an explicit end event. We don't reset the last
                        // observation for this task then. It's going to be
                        // reset when its ending is detected.
                        int last_task = this->task_per_player.at(i)(0, 0)(0, 0);
                        if (last_task == CLEARING_RUBBLE) {
                            this->task_per_player[i] = Tensor3(NO_TASK);
                        }
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
                }
                else if (json_message["data"]["mission_state"] == "Stop") {
                    this->mission_finished = true;
                }
            }
            else if (json_message["header"]["message_type"] == "trial") {
                if (json_message["msg"]["sub_type"] == "start") {
                    try {
                        string trial = json_message["data"]["trial_number"];
                        // Remove first character which is the letter T.
                        this->mission_trial_number = stoi(trial.substr(1));
                    }
                    catch (invalid_argument& exp) {
                        this->mission_trial_number = -1;
                    }

                    this->experiment_id = json_message["msg"]["experiment_id"];
                }
                else if (json_message["msg"]["sub_type"] == "stop") {
                    this->mission_finished = true;
                }
            }
            else if (!this->mission_finished) {
                this->fill_observation(json_message);
            }

            return data;
        } // namespace model

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

                    for (int i = 0; i < this->num_players; i++) {
                        data.add_data(fmt::format("TaskP{}", i + 1),
                                      this->task_per_player.at(i));
                        data.add_data(fmt::format("RoleP{}", i + 1),
                                      this->role_per_player.at(i));

                        // Carrying ans saving a victim are tasks that have an
                        // explicit end event. We don't reset the last
                        // observation for this task then. It's going to be
                        // reset when its ending is detected.
                        int last_task = this->task_per_player.at(i)(0, 0)(0, 0);
                        if (last_task == CLEARING_RUBBLE) {
                            this->task_per_player[i] = Tensor3(NO_TASK);
                        }
                    }

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
            else {
                this->fill_observation(json_message);
            }

            return data;
        }

        void ASISTMultiPlayerMessageConverter::fill_observation(
            const nlohmann::json& json_message) {
            const string& player_name = json_message["data"]["playername"];

            if (!EXISTS(player_name, this->player_name_to_id)) {
                return;
            }

            int player_id = this->player_name_to_id.at(player_name);

            if (json_message["header"]["message_type"] == "event" &&
                json_message["msg"]["sub_type"] == "Event:ToolUsed" &&
                (json_message["data"]["target_block_type"] ==
                     "minecraft:gravel" ||
                 json_message["data"]["target_block_type"] == "Gravel") &&
                json_message["data"]["tool_type"] == "HAMMER") {
                this->task_per_player[player_id] = Tensor3(CLEARING_RUBBLE);
            }
            else if (json_message["header"]["message_type"] == "event" &&
                     json_message["msg"]["sub_type"] == "Event:Triage") {
                if (json_message["data"]["triage_state"] == "IN_PROGRESS") {
                    if (json_message["data"]["color"] == "Green") {
                        this->task_per_player[player_id] =
                            Tensor3(SAVING_REGULAR);
                    }
                    else if (json_message["data"]["color"] == "Yellow") {
                        this->task_per_player[player_id] =
                            Tensor3(SAVING_CRITICAL);
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
                cout << role << endl;
                if (role == "Search_Specialist" || role == "search") {
                    this->role_per_player[player_id] = Tensor3(SEARCH);
                }
                else if (role == "Hazardous_Material_Specialist" ||
                         role == "hammer") {
                    this->role_per_player[player_id] = Tensor3(HAMMER);
                }
                else if (role == "Medical_Specialist" || role == "medical") {
                    this->role_per_player[player_id] = Tensor3(MEDICAL);
                }
                else {
                    this->role_per_player[player_id] = Tensor3(NO_ROLE);
                }
            }
        }

        void ASISTMultiPlayerMessageConverter::prepare_for_new_mission() {
            this->player_name_to_id.clear();
        }

        bool ASISTMultiPlayerMessageConverter::is_valid_message_file(
            const boost::filesystem::directory_entry& file) const {
            const string& filename = file.path().filename().string();
            return filename.find("TrialMessages") != string::npos &&
                   filename.find("Training") == string::npos &&
                   filename.find("PlanningASR") == string::npos &&
                   file.path().extension().string() == ".metadata";
        }

    } // namespace model
} // namespace tomcat
