#include "ASISTMessageConverter.h"

#include <fstream>
#include <iomanip>

#include <fmt/format.h>

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        ASISTMessageConverter::ASISTMessageConverter(int num_seconds,
                                                     int time_step_size)
            : MessageConverter(num_seconds, time_step_size) {}

        ASISTMessageConverter::~ASISTMessageConverter() {}

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        map<string, nlohmann::json>
        ASISTMessageConverter::filter(const string& messages_filepath) const {
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
                        string error_msg = "Invalid format. Some messages do "
                                           "not contain a header timestamp.";
                        throw TomcatModelException(error_msg);
                    }

                    if (json_message["topic"] != nullptr) {
                        const string& topic = json_message["topic"];

                        if (EXISTS(topic, this->get_used_topics())) {
                            const string& timestamp =
                                json_message["header"]["timestamp"];
                            messages[timestamp] = json_message;
                        }
                    }
                }
                catch (nlohmann::detail::parse_error& exp) {
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