#include "TA3MessageConverter.h"

#include <fstream>

#include <fmt/format.h>

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        TA3MessageConverter::TA3MessageConverter(
            int num_seconds,
            int time_step_size,
            const std::string& map_filepath)
            : MessageConverter(num_seconds, time_step_size) {

            this->load_map_area_configuration(map_filepath);
        }

        TA3MessageConverter::~TA3MessageConverter() {}

        //----------------------------------------------------------------------
        // Copy & Move constructors/assignments
        //----------------------------------------------------------------------
        TA3MessageConverter::TA3MessageConverter(
            const TA3MessageConverter& converter) {
            this->copy_converter(converter);
        }

        TA3MessageConverter&
        TA3MessageConverter::operator=(const TA3MessageConverter& converter) {
            this->copy_converter(converter);
            return *this;
        }

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        map<string, nlohmann::json>
        TA3MessageConverter::filter(const string& messages_filepath) const {
            map<string, nlohmann::json> messages;

            ifstream file_reader(messages_filepath);
            while (!file_reader.eof()) {
                string message;
                getline(file_reader, message);
                try {
                    nlohmann::json json_message =
                        nlohmann::json::parse(message);
                    if (!json_message.contains("header") ||
                        !json_message["header"].contains("timestamp")) {
                        throw TomcatModelException("Invalid format. "
                                                   "Some messages do not "
                                                   "contain a header timestamp"
                                                   ".");
                    }

                    const string& topic = json_message["topic"];

                    if (topic == "trial" ||
                        (topic == "observations/events/mission") ||
                        topic == "observations/state" ||
                        topic == "observations/events/player/triage" ||
                        topic == "observations/events/player/location" ||
                        topic == "observations/events/player/beep") {

                        const string& timestamp =
                            json_message["header"]["timestamp"];
                        messages[timestamp] = json_message;
                    }
                }
                catch (nlohmann::detail::parse_error& exp) {
                }
            }

            return messages;
        }

        EvidenceSet TA3MessageConverter::get_data_from_message(
            const nlohmann::json& json_message,
            nlohmann::json& json_mission_log) {

            EvidenceSet data;

            if (this->new_mission) {
                this->mission_started = false;
                this->new_mission = false;
            }

            if (!this->mission_started) {
                if (json_message["topic"] == "observations/events/mission" &&
                    json_message["data"]["mission_state"] == "Start") {

                    this->mission_started = true;
                    this->elapsed_time_steps = 0;

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
                }
            }
            else {
                if (json_message["topic"] == "observations/state") {
                    int elapsed_time = this->get_elapsed_time(
                        json_message["data"]["mission_timer"]);

                    if (elapsed_time ==
                        this->elapsed_time_steps + this->time_step_size) {
                        data.add_data("TrainingCondition",
                                      this->training_condition);
                        data.add_data("Area", this->area);
                        data.add_data("Task", this->task);
                        data.add_data("Beep", this->beep);

                        // Event that happens at a single time step.
                        // Therefore, we must reset the value after saving.
                        this->beep = Tensor3(0);

                        this->elapsed_time_steps += this->time_step_size;

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

        void TA3MessageConverter::load_map_area_configuration(
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

        int TA3MessageConverter::get_elapsed_time(const string& time) {
            int minutes = 0;
            int seconds = 0;

            try {
                minutes = stoi(time.substr(0, time.find(":")));
                seconds = stoi(time.substr(time.find(":") + 1, time.size()));
            }
            catch (invalid_argument& e) {
            }

            return this->time_steps - (seconds + minutes * 60);
        }

        void TA3MessageConverter::fill_victim_saving_observation(
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

        void TA3MessageConverter::fill_room_observation(
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

        void TA3MessageConverter::fill_beep_observation(
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
