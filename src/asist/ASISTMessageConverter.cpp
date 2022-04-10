#include "ASISTMessageConverter.h"

#include <boost/algorithm/string.hpp>
#include <fstream>
#include <iomanip>

#include <fmt/format.h>

namespace tomcat {
    namespace model {

        using namespace std;
        namespace fs = boost::filesystem;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        ASISTMessageConverter::ASISTMessageConverter(int num_seconds,
                                                     int time_step_size)
            : MessageConverter(num_seconds, time_step_size) {}

        ASISTMessageConverter::~ASISTMessageConverter() {}

        //----------------------------------------------------------------------
        // Static functions
        //----------------------------------------------------------------------
        bool
        ASISTMessageConverter::is_message_of(const nlohmann::json& json_message,
                                             const std::string& type) {

            return boost::iequals(
                (string)json_message["header"]["message_type"], type);
        }

        bool
        ASISTMessageConverter::is_message_of(const nlohmann::json& json_message,
                                             const std::string& type,
                                             const std::string& sub_type) {

            return boost::iequals(
                       (string)json_message["header"]["message_type"], type) &&
                   boost::iequals((string)json_message["msg"]["sub_type"],
                                  sub_type);
        }

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        void
        ASISTMessageConverter::convert_messages(const std::string& messages_dir,
                                                const std::string& data_dir) {
            // Store FoV files to be appended to the messages if it comes in a
            // separate file.
            this->fov_filepaths.clear();
            string fov_dir = messages_dir + "/FoV";
            if (fs::exists(fov_dir)) {
                for (const auto& file : fs::directory_iterator(fov_dir)) {
                    string filename = file.path().filename().string();
                    const size_t start = filename.find("T00");
                    const size_t end = filename.find("_Member");
                    string trial_team = filename.substr(start, end - start);
                    this->fov_filepaths[trial_team] = file.path().string();
                }
            }
            MessageConverter::convert_messages(messages_dir, data_dir);
        }

        EvidenceSet ASISTMessageConverter::get_data_from_message(
            const nlohmann::json& json_message,
            nlohmann::json& json_mission_log) {

            if (this->mission_finished) {
                this->mission_started = false;
                this->mission_finished = false;
                this->prepare_for_new_mission();
                json_mission_log["step_size"] = this->time_step_size;
            }

            EvidenceSet data;
            if (!this->mission_started) {
                data = this->parse_before_mission_start(json_message,
                                                        json_mission_log);
            }
            else {
                data = this->parse_after_mission_start(json_message,
                                                       json_mission_log);
            }

            return data;
        }

        void ASISTMessageConverter::parse_individual_message(
            const nlohmann::json& json_message) {}

        map<string, nlohmann::json>
        ASISTMessageConverter::filter(const string& messages_filepath) {
            map<string, nlohmann::json> messages;
            bool contains_fov = false;

            ifstream file_reader(messages_filepath);
            while (!file_reader.eof()) {
                string message;
                getline(file_reader, message);
                try {
                    nlohmann::json json_message =
                        nlohmann::json::parse(message);
                    if (!json_message.contains("msg") ||
                        !json_message["msg"].contains("timestamp")) {
                        string error_msg = "Invalid format. Some messages do "
                                           "not contain a timestamp.";
                        throw TomcatModelException(error_msg);
                    }

                    if (json_message["topic"] != nullptr) {
                        const string& topic = json_message["topic"];

                        if (EXISTS(topic, this->get_used_topics())) {
                            if (topic == "trial") {
                                // There's an issue with the timestamp in the
                                // msg section of trial messages. The
                                // timestamp in this section is not being
                                // updated when the trial stops.
                                const string& timestamp =
                                    json_message["header"]["timestamp"];
                                messages[timestamp] = json_message;
                            }
                            else {
                                const string& timestamp =
                                    json_message["msg"]["timestamp"];
                                messages[timestamp] = json_message;
                            }

                            this->parse_individual_message(json_message);
                        }
                    }
                }
                catch (nlohmann::detail::parse_error& exp) {
                }
            }

            // Look for FoV data in a separate file in the same directory if
            // it's not contained in the main messages file.
            //            if (!contains_fov) {
            const size_t start = messages_filepath.find("T00");
            const size_t end = messages_filepath.find("_Member");
            string trial_team = messages_filepath.substr(start, end - start);

            if (EXISTS(trial_team, this->fov_filepaths)) {
                const string& fov_filepath = this->fov_filepaths[trial_team];
                ifstream fov_file_reader(fov_filepath);
                if (fov_file_reader.good()) {
                    while (!fov_file_reader.eof()) {
                        string message;
                        getline(fov_file_reader, message);
                        try {
                            nlohmann::json json_message =
                                nlohmann::json::parse(message);
                            if (!json_message.contains("msg") ||
                                !json_message["msg"].contains("timestamp")) {
                                string error_msg =
                                    "Invalid format. Some messages do "
                                    "not contain a timestamp.";
                                throw TomcatModelException(error_msg);
                            }

                            const string& timestamp =
                                json_message["msg"]["timestamp"];
                            messages[timestamp] = json_message;

                            this->parse_individual_message(json_message);
                        }
                        catch (nlohmann::detail::parse_error& exp) {
                        }
                    }
                }
            }

            return messages;
        }

        void ASISTMessageConverter::copy_converter(
            const ASISTMessageConverter& converter) {
            MessageConverter::copy_converter(converter);
            this->mission_initial_timestamp =
                converter.mission_initial_timestamp;
            this->mission_trial_number = converter.mission_trial_number;
            this->experiment_id = converter.experiment_id;
        }

        int ASISTMessageConverter::get_elapsed_time(const string& time) {
            int minutes = 0;
            int seconds = 0;

            try {
                minutes = stoi(time.substr(0, time.find(":")));
                seconds = stoi(time.substr(time.find(":") + 1, time.size()));
            }
            catch (invalid_argument& e) {
                return NO_OBS;
            }

            return this->time_steps * this->time_step_size -
                   (seconds + minutes * 60);
        }

        //----------------------------------------------------------------------
        // Getters & Setters
        //----------------------------------------------------------------------
        time_t ASISTMessageConverter::get_mission_initial_timestamp() const {
            return mission_initial_timestamp;
        }

        int ASISTMessageConverter::get_mission_trial_number() const {
            return mission_trial_number;
        }

        const string& ASISTMessageConverter::get_experiment_id() const {
            return experiment_id;
        }

    } // namespace model
} // namespace tomcat
