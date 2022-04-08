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
            this->log_file << "Activate\t";
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

        void ASISTStudy3InterventionLogger::log_intervene_on_introduction(
            int time_step) {
            string text = "Introduction intervention.";
            this->log_trigger_intervention(time_step, text);
        }

        void ASISTStudy3InterventionLogger::log_intervene_on_motivation(
            int time_step, int num_encouragements, double probability) {
            string text = fmt::format(
                "Motivation intervention. p(encouragement < {}) = {}",
                num_encouragements,
                probability);
            this->log_trigger_intervention(time_step, text);
        }

        void ASISTStudy3InterventionLogger::log_watch_motivation_intervention(
            int time_step) {
            string text =
                "Started counting number of encouragement utterances.";
            this->log_watch_intervention(time_step, text);
        }

        void ASISTStudy3InterventionLogger::log_cancel_motivation_intervention(
            int time_step, int num_encouragements, double probability) {
            string text = fmt::format(
                "Motivation intervention. p(encouragement < {}) = {}",
                num_encouragements,
                probability);
            this->log_cancel_intervention(time_step, text);
        }

        void ASISTStudy3InterventionLogger::log_update_motivation_intervention(
            int time_step, int num_encouragements) {
            string text = fmt::format("{} encouragement utterances detected.",
                                      num_encouragements);
            this->log_watch_intervention(time_step, text);
        }

        void
        ASISTStudy3InterventionLogger::log_empty_encouragements(int time_step) {
            string text =
                "No encouragement detected. Agent was probably restarted.";
            this->log_watch_intervention(time_step, text);
        }

        void ASISTStudy3InterventionLogger::log_intervene_on_marker(
            int time_step, int player_order) {

            string text =
                fmt::format("Communication-marker intervention for player {}.",
                            PLAYER_ORDER_TO_COLOR.at(player_order));
            this->log_trigger_intervention(time_step, text);
        }

        void ASISTStudy3InterventionLogger::log_intervene_on_ask_for_help(
            int time_step, int player_order) {

            string text =
                fmt::format("Ask-for-help intervention for player {}.",
                            PLAYER_ORDER_TO_COLOR.at(player_order));
            this->log_trigger_intervention(time_step, text);
        }

        void ASISTStudy3InterventionLogger::
            log_cancel_communication_marker_intervention(
                int time_step,
                int player_order,
                const ASISTStudy3MessageConverter::Marker& marker,
                bool speech,
                bool marker_removal) {

            string text;

            if (speech && marker_removal) {
                text = fmt::format(
                    "{} spoke about and removed {} marker.",
                    PLAYER_ORDER_TO_COLOR.at(player_order),
                    ASISTStudy3MessageConverter::MARKER_TYPE_TO_TEXT.at(
                        marker.type));
            } else if (speech) {
                text = fmt::format(
                    "{} spoke about {}.",
                    PLAYER_ORDER_TO_COLOR.at(player_order),
                    ASISTStudy3MessageConverter::MARKER_TYPE_TO_TEXT.at(
                        marker.type));
            }
            else if (marker_removal) {
                text = fmt::format(
                    "{} removed {} marker.",
                    PLAYER_ORDER_TO_COLOR.at(player_order),
                    ASISTStudy3MessageConverter::MARKER_TYPE_TO_TEXT.at(
                        marker.type));
            }

            this->log_cancel_intervention(time_step, text);
        }


        void ASISTStudy3InterventionLogger::
            log_watch_communication_marker_intervention(
                int time_step,
                int player_order,
                const ASISTStudy3MessageConverter::Marker& marker) {

            string text =
                fmt::format("{} placed {} marker.",
                            PLAYER_ORDER_TO_COLOR.at(player_order),
                            ASISTStudy3MessageConverter::MARKER_TYPE_TO_TEXT.at(
                                marker.type));
            this->log_watch_intervention(time_step, text);
        }

        void ASISTStudy3InterventionLogger::
            log_activate_communication_marker_intervention(
                int time_step,
                int player_order,
                const ASISTStudy3MessageConverter::Marker& active_marker,
                bool area_changed,
                bool victim_interaction,
                bool marker_placed) {

            string text;
            const string& player_color = PLAYER_ORDER_TO_COLOR.at(player_order);
            if (area_changed && victim_interaction) {
                text = fmt::format("{} changed area and interacted with "
                                   "victim",
                                   player_color);
            }
            else if (area_changed && marker_placed) {
                text = fmt::format("{} changed area and placed a new marker",
                                   player_color);
            }
            else if (area_changed) {
                text = fmt::format("{} changed area", player_color);
            }
            else if (victim_interaction) {
                text = fmt::format("{} interacted with victim", player_color);
            }
            else if (marker_placed) {
                text = fmt::format("{} placed a marker", player_color);
            }

            text =
                fmt::format("{}. {} marker gets ready for intervention.",
                            text,
                            ASISTStudy3MessageConverter::MARKER_TYPE_TO_TEXT.at(
                                active_marker.type));

            this->log_activate_intervention(time_step, text);
        }

        void ASISTStudy3InterventionLogger::log_watch_ask_for_help_intervention(
            int time_step, int player_order) {
            string text = fmt::format("{} needs help.",
                                      PLAYER_ORDER_TO_COLOR.at(player_order));
            this->log_watch_intervention(time_step, text);
        }

        void
        ASISTStudy3InterventionLogger::log_activate_ask_for_help_intervention(
            int time_step, int player_order, int latency) {
            string text =
                fmt::format("{} did not ask for help in the last {} seconds.",
                            PLAYER_ORDER_TO_COLOR.at(player_order),
                            latency);
            this->log_activate_intervention(time_step, text);
        }

        void
        ASISTStudy3InterventionLogger::log_cancel_ask_for_help_intervention(
            int time_step,
            int player_order,
            bool area_changed,
            bool help_requested) {
            string text;

            if (area_changed) {
                fmt::format("{} changed area.",
                            PLAYER_ORDER_TO_COLOR.at(player_order));
            }
            else if (help_requested) {
                fmt::format("{} asked for help.",
                            PLAYER_ORDER_TO_COLOR.at(player_order));
            }

            text =
                fmt::format("{}. Canceling ask-for-help intervention.", text);

            this->log_cancel_intervention(time_step, text);
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

        void ASISTStudy3InterventionLogger::log_activate_intervention(
            int time_step, const string& text) {
            this->log_common_info(time_step_to_mission_timer(time_step), text);
            this->log_file << "\t\t"; // watch and cancel
            this->log_file << "X";    // activate
            this->log_file << "\n";
            this->log_file.flush();
        }

        void ASISTStudy3InterventionLogger::log_trigger_intervention(
            int time_step, const string& text) {
            this->log_common_info(time_step_to_mission_timer(time_step), text);
            this->log_file << "\t\t\t"; // watch, cancel, activate
            this->log_file << "X";      // intervene
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
