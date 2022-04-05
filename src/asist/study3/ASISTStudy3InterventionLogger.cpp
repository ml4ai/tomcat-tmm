#include "ASISTStudy3InterventionLogger.h"

#include <iomanip>

#include <fmt/format.h>

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        ASISTStudy3InterventionLogger::ASISTStudy3InterventionLogger(
            const string& log_filepath)
            : OnlineLogger(change_extension(log_filepath)) {}

        //----------------------------------------------------------------------
        // Static functions
        //----------------------------------------------------------------------
        string ASISTStudy3InterventionLogger::change_extension(
            const string& filepath) {
            // This logger logs info in a TSV format. We change the extension of
            // the file to reflect that.
            return filepath.substr(0, filepath.find_last_of('.')) + ".tsv";
        }

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        void ASISTStudy3InterventionLogger::create_header() {
            this->log_file << "Timestamp\t";
            this->log_file << "Mission Timer\t";
            this->log_file << "Text\t";
            this->log_file << "Watch\t";
            this->log_file << "Cancel\t";
            this->log_file << "Intervene\t";
            this->log_file << "\n";
        }

        void ASISTStudy3InterventionLogger::log(const string& text) {
            if (!this->initialized) {
                this->create_header();
                this->initialized = true;
            }

            this->log_common_info("", text);
            this->log_file << "\n";
            this->log_file.flush();
        }

        void ASISTStudy3InterventionLogger::log_first_evidence_set(
            const EvidenceSet& data) {
            int mission_order = data.get_metadata()[0]["mission_order"];
            const string& trial = data.get_metadata()[0]["trial"];
            const string& experiment_id =
                data.get_metadata()[0]["experiment_id"];
            this->log(fmt::format(
                "Mission {} from trial {} and experiment id {} started.",
                mission_order,
                trial,
                experiment_id));
        }

        void ASISTStudy3InterventionLogger::log_watch_intervention(
            int time_step, const string& text) {
            this->log_common_info(time_step_to_mission_timer(time_step), text);
            this->log_file << "X\t";
            this->log_file << "\n";
            this->log_file.flush();
        }

        void ASISTStudy3InterventionLogger::log_cancel_intervention(
            int time_step, const string& text) {
            this->log_common_info(time_step_to_mission_timer(time_step), text);
            this->log_file << "\t";  // watch
            this->log_file << "X\t"; // cancel
            this->log_file << "\n";
            this->log_file.flush();
        }

        void ASISTStudy3InterventionLogger::log_trigger_intervention(
            int time_step, const string& text) {
            this->log_common_info(time_step_to_mission_timer(time_step), text);
            this->log_file << "\t\t"; // watch and cancel
            this->log_file << "X";    // intervene
            this->log_file << "\n";
            this->log_file.flush();
        }

        void ASISTStudy3InterventionLogger::log(int time_step,
                                                const string& text) {
            this->log_common_info(time_step_to_mission_timer(time_step), text);
            this->log_file << "\n";
            this->log_file.flush();
        }

        void ASISTStudy3InterventionLogger::log_common_info(
            const string& mission_timer, const string& text) {
            this->log_file << Timer::get_current_timestamp() << "\t";
            this->log_file << mission_timer << "\t";
            this->log_file << text << "\t";
        }

        std::string ASISTStudy3InterventionLogger::time_step_to_mission_timer(
            int time_step) const {

            int seconds =
                this->num_time_steps - time_step * this->time_step_size;
            int minutes = seconds / 60;
            seconds = seconds % 60;

            string timer;
            if (minutes <= 9) {
                timer = "0" + to_string(minutes);
            }
            else {
                timer = to_string(minutes);
            }

            if (seconds <= 9) {
                timer = timer + ":0" + to_string(seconds);
            }
            else {
                timer = timer + ":" + to_string(seconds);
            }

            return timer;
        }

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        void ASISTStudy3InterventionLogger::set_time_step_size(
            int new_time_step_size) {
            this->time_step_size = new_time_step_size;
        }
        void ASISTStudy3InterventionLogger::set_num_time_steps(
            int new_num_time_steps) {
            this->num_time_steps = new_num_time_steps;
        }

    } // namespace model
} // namespace tomcat
