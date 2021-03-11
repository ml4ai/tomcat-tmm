#include "ASISTSinglePlayerMessageConverter.h"

#include <fstream>
#include <iomanip>

#include <fmt/format.h>

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        ASISTSinglePlayerMessageConverter::ASISTSinglePlayerMessageConverter(
            int num_seconds,
            int time_step_size,
            const std::string& map_filepath)
            : ASISTMessageConverter(num_seconds, time_step_size) {

            this->load_map_area_configuration(map_filepath);
        }

        ASISTSinglePlayerMessageConverter::
            ~ASISTSinglePlayerMessageConverter() {}

        //----------------------------------------------------------------------
        // Copy & Move constructors/assignments
        //----------------------------------------------------------------------
        ASISTSinglePlayerMessageConverter::ASISTSinglePlayerMessageConverter(
            const ASISTSinglePlayerMessageConverter& converter)
            : ASISTMessageConverter(converter.time_steps *
                                        converter.time_step_size,
                                    converter.time_step_size) {
            this->copy_converter(converter);
        }

        ASISTSinglePlayerMessageConverter&
        ASISTSinglePlayerMessageConverter::operator=(
            const ASISTSinglePlayerMessageConverter& converter) {
            this->copy_converter(converter);
            return *this;
        }

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        void ASISTSinglePlayerMessageConverter::copy_converter(
            const ASISTSinglePlayerMessageConverter& converter) {
            ASISTMessageConverter::copy_converter(converter);
            this->mission_started = converter.mission_started;
            this->training_condition = converter.training_condition;
            this->area = converter.area;
            this->task = converter.task;
            this->beep = converter.beep;
            this->elapsed_time = converter.elapsed_time;
            this->map_area_configuration = converter.map_area_configuration;
        }

        unordered_set<string>
        ASISTSinglePlayerMessageConverter::get_used_topics() const {
            unordered_set<string> topics;

            topics.insert("trial");
            topics.insert("observations/events/mission");
            topics.insert("observations/state");
            topics.insert("observations/events/player/triage");
            topics.insert("observations/events/player/location");
            topics.insert("observations/events/player/beep");

            return topics;
        }

        EvidenceSet ASISTSinglePlayerMessageConverter::get_data_from_message(
            const nlohmann::json& json_message,
            nlohmann::json& json_mission_log) {

            EvidenceSet data;

            if (this->mission_finished) {
                this->mission_started = false;
                this->mission_finished = false;
            }

            if (!this->mission_started) {
                if (json_message["topic"] == "observations/events/mission" &&
                    json_message["data"]["mission_state"] == "Start") {

                    this->mission_started = true;
                    this->elapsed_time = this->time_step_size;

                    if (this->training_condition.is_empty()) {
                        // In case the message for training condition arrives
                        // later, this will set the training condition as non
                        // observed at time 0.
                        this->training_condition = Tensor3(NO_OBS);
                    }

                    data.add_data("TrainingCondition",
                                  this->training_condition);
                    data.add_data("Area", Tensor3(NO_OBS));
                    data.add_data("Task", Tensor3(NO_OBS));
                    data.add_data("Beep", Tensor3(NO_OBS));

                    this->area = Tensor3(0);
                    this->task = Tensor3(0);
                    this->beep = Tensor3(0);

                    // Add initial timestamp to the log
                    const string& timestamp =
                        json_message["header"]["timestamp"];
                    json_mission_log["initial_timestamp"] = timestamp;

                    tm t{};
                    istringstream ss(timestamp);
                    // The precision of the timestamp will be in seconds.
                    // milliseconds are ignored. This can be reaccessed later
                    // if necessary. The milliseconds could be stored in a
                    // separate attribute of this class.
                    ss >> get_time(&t, "%Y-%m-%dT%T");
                    if (!ss.fail()) {
                        this->mission_initial_timestamp = mktime(&t);
                    }
                }
                else if (json_message["topic"] == "trial") {
                    int value;
                    try {
                        value =
                            stoi((string)json_message["data"]["condition"]) - 1;

                        if (value <= 2) {
                            this->training_condition = Tensor3(value);
                        }
                        else {
                            throw TomcatModelException(
                                "Training condition > 2.");
                        }
                    }
                    catch (invalid_argument& exp) {
                        throw TomcatModelException(fmt::format(
                            "Invalid training condition {}.", value));
                    }

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
            }
            else {
                if (json_message["topic"] == "observations/state") {
                    const string& timer = json_message["data"]["mission_timer"];
                    int elapsed_time = this->get_elapsed_time(timer);

                    if (elapsed_time ==
                        this->elapsed_time + this->time_step_size) {
                        data.add_data("TrainingCondition",
                                      this->training_condition);
                        data.add_data("Area", this->area);
                        data.add_data("Task", this->task);
                        data.add_data("Beep", this->beep);

                        // Event that happens at a single time step.
                        // Therefore, we must reset the value after saving.
                        this->beep = Tensor3(0);

                        // If the player starts to rescue a yellow victim close
                        // to the time limit to rescue this kind of victim (half
                        // of the mission total time) and he doesn't finish
                        // before this limit, there's no SUCCESSFUL or
                        // UNSUCCESSFUL message reported by the testbed. In that
                        // case, we need to reset the observation manually
                        // here, otherwise, an observation for yellow (value 2)
                        // will be emitted until the end of the game.
                        if (elapsed_time > this->time_steps / 2 &&
                            ((int)this->task.at(0, 0, 0)) == 2) {
                            this->task = Tensor3(0);
                        }

                        this->elapsed_time += this->time_step_size;
                        if (this->elapsed_time ==
                            this->time_steps * this->time_step_size) {
                            this->mission_finished = true;
                        }
                    }
                }
                else if (json_message["topic"] ==
                         "observations/events/player/triage") {
                    this->fill_victim_saving_observation(json_message);
                }
                else if (json_message["topic"] ==
                         "observations/events/player/location") {
                    this->fill_room_observation(json_message);
                }
                else if (json_message["topic"] ==
                         "observations/events/player/beep") {
                    this->fill_beep_observation(json_message);
                }
            }

            return data;
        }

        bool ASISTSinglePlayerMessageConverter::is_valid_message_file(
            const boost::filesystem::directory_entry& file) const {
            const string& filename = file.path().filename().string();
            return filename.find("HSRData_TrialMessages") != string::npos &&
                   filename.find("Vers-3") != string::npos &&
                   file.path().extension().string() == ".metadata";
        }

        void ASISTSinglePlayerMessageConverter::load_map_area_configuration(
            const string& map_filepath) {

            fstream map_file;
            map_file.open(map_filepath);
            if (map_file.is_open()) {
                nlohmann::json json_map = nlohmann::json::parse(map_file);
                for (const auto& location : json_map["locations"]) {
                    const string area_type = location["type"];
                    this->map_area_configuration[location["id"]] =
                        area_type.find("room") != string::npos;
                }
            }
            else {
                stringstream ss;
                ss << "Map configuration file in " << map_filepath
                   << " does not exist.";
                throw TomcatModelException(ss.str());
            }
        }

        void ASISTSinglePlayerMessageConverter::fill_victim_saving_observation(
            const nlohmann::json& json_message) {

            if (json_message["data"]["triage_state"] == "IN_PROGRESS") {
                if (json_message["data"]["color"] == "Green") {
                    this->task = Tensor3(1);
                }
                else if (json_message["data"]["color"] == "Yellow") {
                    this->task = Tensor3(2);
                }
            }
            else {
                this->task = Tensor3(0);
            }
        }

        void ASISTSinglePlayerMessageConverter::fill_room_observation(
            const nlohmann::json& json_message) {

            if (json_message["data"].contains("entered_area_id")) {
                // Old version of the location monitor. Do not convert.
                throw TomcatModelException(
                    "Old version of the location monitor.");
            }

            if (json_message["data"].contains("locations")) {
                string room_id = json_message["data"]["locations"][0]["id"];
                if (EXISTS(room_id, this->map_area_configuration)) {
                    if (this->map_area_configuration.at(room_id)) {
                        this->area = Tensor3(1);
                    }
                    else {
                        this->area = Tensor3(0);
                    }
                }
            }
        }

        void ASISTSinglePlayerMessageConverter::fill_beep_observation(
            const nlohmann::json& json_message) {

            const string beep = json_message["data"]["message"];
            if (beep == "Beep") {
                this->beep = Tensor3(1);
            }
            else if (beep == "Beep Beep") {
                this->beep = Tensor3(2);
            }
        }

    } // namespace model
} // namespace tomcat
