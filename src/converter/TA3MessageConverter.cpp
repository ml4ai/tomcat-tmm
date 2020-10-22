#include "TA3MessageConverter.h"

#include <iomanip>
#include <iostream>

#include <boost/progress.hpp>
#include <boost/filesystem.hpp>
#include <fmt/format.h>

#include "utils/EigenExtensions.h"
#include "utils/FileHandler.h"

using namespace std;
namespace fs = boost::filesystem;

namespace tomcat {
    namespace model {

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        TA3MessageConverter::TA3MessageConverter(
            const string& map_config_filepath, int time_gap)
            : MessageConverter(time_gap) {
            this->init_observations();
            this->load_map_area_configuration(map_config_filepath);
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
        void TA3MessageConverter::init_observations() {
            this->last_observations_per_node[ROOM] = NO_OBS;
            this->last_observations_per_node[SG] = NO_OBS;
            this->last_observations_per_node[SY] = NO_OBS;
            this->last_observations_per_node[Q] = NO_OBS;
            this->last_observations_per_node[BEEP] = NO_OBS;
        }

        void TA3MessageConverter::load_map_area_configuration(
            const string& map_config_filepath) {
            fstream map_config_file;
            map_config_file.open(map_config_filepath);
            if (map_config_file.is_open()) {
                nlohmann::json json_map_config =
                    nlohmann::json::parse(map_config_file);
                for (const auto& location : json_map_config["locations"]) {
                    const string area_type = location["type"];
                    this->map_area_configuration[location["id"]] =
                        area_type.find("room") != string::npos;
                }
            }
            else {
                stringstream ss;
                ss << "Map configuration file in " << map_config_filepath
                   << " does not exist.";
                throw TomcatModelException(ss.str());
            }
        }

        void TA3MessageConverter::convert_offline(const string& input_dir,
                                                  const string& output_dir) {

            nlohmann::json json_metadata = this->get_metadata(output_dir);
            unordered_set<string> processed_filenames =
                this->get_processed_message_filenames(json_metadata);
            vector<fs::path> message_filepaths =
                this->get_message_filepaths(input_dir, processed_filenames);
            int num_message_files = message_filepaths.size();
            boost::progress_display progress(num_message_files);

            int d = 1;
            unordered_map<string, Eigen::MatrixXd> observations_per_node;
            for (const auto& filepath : message_filepaths) {

                vector<nlohmann::json> messages =
                    this->get_sorted_messages_in(filepath.string());

                this->training_condition = NO_OBS;
                this->time_step = 0;
                this->mission_started = false;
                this->init_observations();

                try {
                    for (auto& message : messages) {
                        for (const auto& [node_label, value] :
                             this->convert_online(message)) {

                            // Initialize matrix of observations with
                            // non observed value
                            if (!EXISTS(node_label, observations_per_node)) {
                                observations_per_node[node_label] =
                                    Eigen::MatrixXd::Constant(1, T + 1, NO_OBS);
                            }

                            // Append one more row for a new mission
                            // trial
                            if (observations_per_node[node_label].rows() ==
                                d - 1) {
                                observations_per_node[node_label]
                                    .conservativeResize(d, Eigen::NoChange);
                                observations_per_node[node_label].row(d - 1) =
                                    Eigen::MatrixXd::Constant(1, T + 1, NO_OBS);
                            }

                            if (this->time_step <= T) {
                                observations_per_node[node_label](
                                    d - 1, this->time_step) = value;
                            }
                        }

                        if (this->time_step > T) {
                            break;
                        }
                    }

                    if (this->time_step < T) {
                        // The mission ended before the total amount of
                        // seconds expected. Discard this mission trial
                        // and emit a message.
                        throw TomcatModelException("Early stopping.");
                    }
                    else {
                        d++;

                        // Add file to the converted files metadata
                        nlohmann::json json_message_file;
                        json_message_file["name"] =
                            filepath.filename().string();
                        json_message_file["initial_timestamp"] =
                            this->initial_timestamp;
                        json_metadata["files_converted"].push_back(
                            json_message_file);
                    }
                }
                catch (TomcatModelException& exp) {
                    nlohmann::json json_message_file;
                    json_message_file["name"] = filepath.filename().string();
                    json_message_file["error"] = exp.message;
                    json_metadata["files_not_converted"].push_back(
                        json_message_file);

                    // Remove the last observations from the matrices.
                    for (auto& [node_label, data_matrix] :
                         observations_per_node) {
                        data_matrix.conservativeResize(d - 1, Eigen::NoChange);
                    }
                }

                ++progress;
            }

            boost::filesystem::create_directories(output_dir);
            this->save_metadata(json_metadata, output_dir);
            this->merge_and_save_observations(observations_per_node,
                                              output_dir);
        }

        nlohmann::json
        TA3MessageConverter::get_metadata(const string& metadata_dir) {
            // We store the files processed in a metadata file so that new
            // conversions can skip messages already converted.
            nlohmann::json json_metadata;
            const string metadata_filepath =
                get_filepath(metadata_dir, METADATA_FILENAME);
            ifstream metadata_file_reader(metadata_filepath);
            if (metadata_file_reader) {
                json_metadata = nlohmann::json::parse(metadata_file_reader);
            }
            else {
                // If no metadata file exists in the output directory, we create
                // a new json object that will be store the metadata content of
                // the current conversion.
                json_metadata["files_converted"] = nlohmann::json::array();
                json_metadata["files_not_converted"] = nlohmann::json::array();
            }

            return json_metadata;
        }

        unordered_set<string>
        TA3MessageConverter::get_processed_message_filenames(
            const nlohmann::json& json_metadata, bool all) {

            unordered_set<string> filenames;
            for (auto file_prop : json_metadata["files_converted"]) {
                string filename = file_prop["name"];
                filenames.insert(filename);
            }

            if (all) {
                for (auto file_prop : json_metadata["files_not_converted"]) {
                    string filename = file_prop["name"];
                    filenames.insert(filename);
                }
            }

            return filenames;
        }

        vector<boost::filesystem::path>
        TA3MessageConverter::get_message_filepaths(
            const string& messages_dir,
            const std::unordered_set<string>& processed_message_filenames) {

            vector<fs::path> filepaths;
            for (const auto& file : fs::directory_iterator(messages_dir)) {
                string filename = file.path().filename().string();
                if (fs::is_regular_file(file) &&
                    filename.find("TrialMessages") != string::npos) {

                    if (!EXISTS(filename, processed_message_filenames)) {
                        filepaths.push_back(file.path());
                    }
                }
            }

            return filepaths;
        }

        vector<nlohmann::json>
        TA3MessageConverter::get_sorted_messages_in(const string& filepath) {
            vector<nlohmann::json> messages;

            ifstream file_reader(filepath);
            while (!file_reader.eof()) {
                string message;
                getline(file_reader, message);
                try {
                    messages.push_back(nlohmann::json::parse(message));
                } catch (nlohmann::detail::parse_error& exp) {

                }
            }

            sort(messages.begin(),
                 messages.end(),
                 [](const nlohmann::json& lhs, const nlohmann::json& rhs) {
                     return lhs["header"]["timestamp"].dump() <
                            rhs["header"]["timestamp"].dump();
                 });

            return messages;
        }

        unordered_map<string, double>
        TA3MessageConverter::convert_online(const nlohmann::json& message) {
            unordered_map<string, double> observations_per_node;

            if (!this->mission_started) {
                if (message["topic"] == "observations/events/mission" &&
                    message["data"]["mission_state"] == "Start") {

                    this->mission_started = true;
                    this->initial_timestamp = message["header"]["timestamp"];
                    observations_per_node = this->last_observations_per_node;
                }
                else if (message["topic"] == "trial") {
                    const string value = message["data"]["condition"];
                    try {
                        this->training_condition = stoi(value) - 1;

                        if (this->training_condition > 2) {
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

            if (this->mission_started) {
                if (message["topic"] == "observations/state") {
                    int new_time_step =
                        T - this->get_remaining_seconds_from(
                                message["data"]["mission_timer"]);

                    if (new_time_step == this->time_step + this->time_gap) {
                        // The nodes below are not observed in time step zero.
                        // Starting from time step 1, they always emmit a valid
                        // value until the mission ends (0 if no specific value
                        // was detected).
                        if (new_time_step == 1) {
                            this->last_observations_per_node[ROOM] = 0;
                            this->last_observations_per_node[SG] = 0;
                            this->last_observations_per_node[SY] = 0;
                            this->last_observations_per_node[BEEP] = 0;
                            this->last_observations_per_node[Q] =
                                this->training_condition;
                        }

                        observations_per_node =
                            this->last_observations_per_node;

                        // The following nodes are events and should have value
                        // only if an explicit message is received. Others will
                        // preserve the values of the previous observation until
                        // a new message comes to change that.
                        this->last_observations_per_node[BEEP] = 0;
                        this->time_step = new_time_step;

                        // If the player starts to rescue a yellow victim close
                        // to the time limit to rescue this kind of victim (half
                        // of the mission total time) and he doesn't finish
                        // before this limit, there's no SUCCESSFUL or
                        // UNSUCCESSFUL message reported by the testbed. I that
                        // case, we need to reset the observation manually
                        // here, otherwise, a true observation (value 1) will be
                        // emitted until the end of the game.
                        if (new_time_step > T / 2) {
                            this->last_observations_per_node[SY] = 0;
                        }
                    }
                }
                else if (message["topic"] ==
                         "observations/events/player/triage") {
                    this->fill_victim_saving_observation(message);
                }
                else if (message["topic"] ==
                         "observations/events/player/location") {
                    this->fill_room_observation(message);
                }
                else if (message["topic"] ==
                         "observations/events/player/beep") {
                    this->fill_beep_observation(message);
                }
            }

            return observations_per_node;
        }

        int
        TA3MessageConverter::get_remaining_seconds_from(const string& time) {
            int minutes = 0;
            int seconds = 0;

            try {
                minutes = stoi(time.substr(0, time.find(":")));
                seconds = stoi(time.substr(time.find(":") + 1, time.size()));
            }
            catch (invalid_argument& e) {
            }

            return seconds + minutes * 60;
        }

        void TA3MessageConverter::fill_victim_saving_observation(
            const nlohmann::json& json_message) {

            string node_label;
            double value = 0;

            if (json_message["data"]["triage_state"] == "IN_PROGRESS") {
                value = 1;
            }

            if (json_message["data"]["color"] == "Green") {
                node_label = SG;
            }
            else if (json_message["data"]["color"] == "Yellow") {
                node_label = SY;
            }

            this->last_observations_per_node[node_label] = value;
        }

        void TA3MessageConverter::fill_room_observation(
            const nlohmann::json& json_message) {

            int value = 0;
            if (json_message["data"].contains("locations")) {
                string room_id = json_message["data"]["locations"][0]["id"];
                if (EXISTS(room_id, this->map_area_configuration)) {
                    if (this->map_area_configuration.at(room_id)) {
                        value = 1;
                    }

                    this->last_observations_per_node[ROOM] = value;
                }
            }
        }

        void TA3MessageConverter::fill_beep_observation(
            const nlohmann::json& json_message) {

            const string beep = json_message["data"]["message"];
            int value = 1; // Beep. 0 is reserved for not observing a beep.
            if (beep == "Beep Beep") {
                value = 2;
            }

            this->last_observations_per_node[BEEP] = value;
        }

        void
        TA3MessageConverter::save_metadata(const nlohmann::json& json_metadata,
                                           const string& output_dir) {

            string metadata_filepath =
                get_filepath(output_dir, METADATA_FILENAME);
            ofstream output_file;
            output_file.open(metadata_filepath);
            output_file << setw(4) << json_metadata;
        }

        void TA3MessageConverter::merge_and_save_observations(
            const unordered_map<string, Eigen::MatrixXd>& observations_per_node,
            const string& output_dir) {

            for (const auto& [node_label, data_matrix] :
                 observations_per_node) {
                string obs_filepath = get_filepath(output_dir, node_label);
                Eigen::MatrixXd full_data = read_matrix_from_file(obs_filepath);
                vstack(full_data, data_matrix);
                save_matrix_to_file(obs_filepath, full_data);
            }
        }

    } // namespace model
} // namespace tomcat
