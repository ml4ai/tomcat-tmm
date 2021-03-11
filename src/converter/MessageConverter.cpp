#include "MessageConverter.h"

#include <iomanip>

#include <boost/progress.hpp>
#include <unordered_set>

#include "utils/FileHandler.h"

using namespace std;
namespace fs = boost::filesystem;

namespace tomcat {
    namespace model {

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        MessageConverter::MessageConverter() {}

        MessageConverter::MessageConverter(int num_seconds, int time_step_size)
            : time_step_size(time_step_size) {
            this->time_steps = num_seconds / time_step_size;
        }

        MessageConverter::~MessageConverter() {}

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        void
        MessageConverter::copy_converter(const MessageConverter& converter) {
            this->time_step_size = converter.time_step_size;
            this->time_steps = converter.time_steps;
        }

        void MessageConverter::convert_messages(const string& messages_dir,
                                                const string& data_dir) {

            unordered_set<string> unprocessed_filenames =
                this->get_unprocessed_message_filenames(messages_dir, data_dir);
            int num_files = unprocessed_filenames.size();
            cout << "Converting " << num_files << " message files...";
            boost::progress_display progress(num_files);

            nlohmann::json json_log;

            EvidenceSet data;
            for (const auto& mission_filename : unprocessed_filenames) {
                nlohmann::json json_mission_log;

                try {
                    const string filepath =
                        get_filepath(messages_dir, mission_filename);

                    // Retrieve only relevant observations sorted by timestamp
                    map<string, nlohmann::json> messages =
                        this->filter(filepath);

                    int next_time_step = 0;
                    EvidenceSet mission_data;
                    this->start_new_mission();
                    for (const auto& [timestamp, message] : messages) {
                        EvidenceSet new_data = this->get_data_from_message(
                            message, json_mission_log);
                        if (!new_data.empty()) {
                            mission_data.hstack(new_data);
                            next_time_step += 1;

                            // mission_finished can be set to true in
                            // get_data_from_message if the maximum number of
                            // time steps was reached.
                            if (next_time_step >= this->time_steps ||
                                this->mission_finished) {
                                break;
                            }
                        }
                    }

                    if (next_time_step < this->time_steps) {
                        // The mission ended before the total amount of seconds
                        // expected.
                        stringstream ss;
                        ss << "Early stopping at "
                           << next_time_step - this->time_step_size << ".";
                        throw TomcatModelException(ss.str());
                    }

                    data.vstack(mission_data);

                    json_mission_log["name"] = mission_filename;
                    json_log["files_converted"].push_back(json_mission_log);
                }
                catch (TomcatModelException& exp) {
                    json_mission_log["name"] = mission_filename;
                    json_mission_log["error"] = exp.message;
                    json_log["files_not_converted"].push_back(json_mission_log);
                }

                ++progress;
            }

            boost::filesystem::create_directories(data_dir);
            if (!unprocessed_filenames.empty()) {
                string log_filepath =
                    get_filepath(data_dir, "conversion_log.json");
                ofstream log_file;
                log_file.open(log_filepath);
                log_file << setw(4) << json_log;

                data.save(data_dir);
            }
        }

        unordered_set<string>
        MessageConverter::get_unprocessed_message_filenames(
            const string& messages_dir, const string& data_dir) {

            unordered_set<string> processed_files;

            const string metadata_filepath =
                get_filepath(data_dir, "conversion_log.json");
            fstream file;
            file.open(metadata_filepath);
            if (file.is_open()) {
                nlohmann::json json_log = nlohmann::json::parse(file);

                // Files successfully converted
                for (const auto& json_file : json_log["files_converted"]) {
                    processed_files.insert((string)json_file["name"]);
                }

                // Files unsuccessfully converted
                for (const auto& json_file : json_log["files_not_converted"]) {
                    processed_files.insert((string)json_file["name"]);
                }
            }

            // Get files in the message directory that were not previously
            // processed
            unordered_set<string> unprocessed_files;
            for (const auto& file : fs::directory_iterator(messages_dir)) {
                string filename = file.path().filename().string();
                if ((fs::is_regular_file(file)) &&
                    this->is_valid_message_file(file)) {

                    if (!EXISTS(filename, processed_files)) {
                        unprocessed_files.insert(filename);
                    }
                }
            }

            return unprocessed_files;
        }

        void MessageConverter::start_new_mission() { this->mission_finished = true; }

        int MessageConverter::get_time_step_size() const {
            return time_step_size;
        }

        bool MessageConverter::is_mission_finished() const {
            return mission_finished;
        }

    } // namespace model
} // namespace tomcat
