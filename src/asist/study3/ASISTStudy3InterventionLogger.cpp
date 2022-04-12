#include "ASISTStudy3InterventionLogger.h"

#include <iomanip>

#include <boost/algorithm/string.hpp>
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

        void ASISTStudy3InterventionLogger::
            log_cancel_communication_marker_intervention(
                int time_step,
                int player_order,
                const ASISTStudy3MessageConverter::Marker& marker,
                bool speech,
                bool marker_removal) {

            string text;

            if (speech) {
                text = fmt::format(
                    "{} spoke about {}. ",
                    PLAYER_ORDER_TO_COLOR.at(player_order),
                    ASISTStudy3MessageConverter::MARKER_TYPE_TO_TEXT.at(
                        marker.type));
            }
            if (marker_removal) {
                text += fmt::format(
                    "{} removed {} marker. ",
                    PLAYER_ORDER_TO_COLOR.at(player_order),
                    ASISTStudy3MessageConverter::MARKER_TYPE_TO_TEXT.at(
                        marker.type));
            }

            text = fmt::format("{}Canceling communication marker intervention.",
                               text);

            this->log_cancel_intervention(time_step, text);
        }

        void ASISTStudy3InterventionLogger::
            log_hinder_communication_marker_intervention(
                int time_step,
                int player_order,
                const ASISTStudy3MessageConverter::Marker& marker) {

            string text =
                fmt::format("{} placed {} marker but spoke about it recently. "
                            "No need to watch marker for intervention.",
                            PLAYER_ORDER_TO_COLOR.at(player_order),
                            ASISTStudy3MessageConverter::MARKER_TYPE_TO_TEXT.at(
                                marker.type));

            this->log(time_step, text);
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
            if (area_changed) {
                text = fmt::format("{} changed area. ", player_color);
            }
            if (victim_interaction) {
                text +=
                    fmt::format("{} interacted with victim. ", player_color);
            }
            if (marker_placed) {
                text += fmt::format("{} placed a new marker. ", player_color);
            }

            text =
                fmt::format("{}Previous {} marker gets ready for intervention.",
                            text,
                            ASISTStudy3MessageConverter::MARKER_TYPE_TO_TEXT.at(
                                active_marker.type));

            this->log_activate_intervention(time_step, text);
        }

        void ASISTStudy3InterventionLogger::
            log_intervene_on_ask_for_help_critical_victim(int time_step,
                                                          int player_order) {

            string text = fmt::format(
                "Ask-for-help (critical_victim) intervention for player {}.",
                PLAYER_ORDER_TO_COLOR.at(player_order));
            this->log_trigger_intervention(time_step, text);
        }

        void ASISTStudy3InterventionLogger::
            log_watch_ask_for_help_critical_victim_intervention(
                int time_step, int player_order) {

            string text = fmt::format("{} needs help to wake critical victim.",
                                      PLAYER_ORDER_TO_COLOR.at(player_order));
            this->log_watch_intervention(time_step, text);
        }

        void ASISTStudy3InterventionLogger::
            log_activate_ask_for_help_critical_victim_intervention(
                int time_step, int player_order, int latency) {

            string text =
                fmt::format("{} did not ask for help to wake critical "
                            "victim in the last {} seconds.",
                            PLAYER_ORDER_TO_COLOR.at(player_order),
                            latency);
            this->log_activate_intervention(time_step, text);
        }

        void ASISTStudy3InterventionLogger::
            log_cancel_ask_for_help_critical_victim_intervention(
                int time_step,
                int player_order,
                bool area_changed,
                bool help_requested,
                bool mention_to_critical_victim,
                bool other_players_around) {
            string text;

            if (area_changed) {
                fmt::format("{} changed area. ",
                            PLAYER_ORDER_TO_COLOR.at(player_order));
            }
            if (help_requested) {
                text += fmt::format("{} asked for help. ",
                                    PLAYER_ORDER_TO_COLOR.at(player_order));
            }
            if (mention_to_critical_victim) {
                text += fmt::format("{} mentioned critical victim. ",
                                    PLAYER_ORDER_TO_COLOR.at(player_order));
            }
            if (other_players_around) {
                text += fmt::format(
                    "At least one other player is in the same location as {}. ",
                    PLAYER_ORDER_TO_COLOR.at(player_order));
            }

            text = fmt::format(
                "{}Canceling ask-for-help (critical victim) intervention.",
                text);

            this->log_cancel_intervention(time_step, text);
        }

        void ASISTStudy3InterventionLogger::
            log_hinder_ask_for_help_critical_victim_intervention(
                int time_step, int player_order) {
            string text = fmt::format(
                "{} needs help to wake critical victim but mentioned that "
                "recently. No need to watch for intervention.",
                PLAYER_ORDER_TO_COLOR.at(player_order));

            this->log(time_step, text);
        }

        void
        ASISTStudy3InterventionLogger::log_intervene_on_ask_for_help_threat(
            int time_step, int player_order) {

            string text =
                fmt::format("Ask-for-help (threat) intervention for player {}.",
                            PLAYER_ORDER_TO_COLOR.at(player_order));
            this->log_trigger_intervention(time_step, text);
        }

        void ASISTStudy3InterventionLogger::
            log_watch_ask_for_help_threat_intervention(int time_step,
                                                       int player_order) {

            string text = fmt::format("{} needs help to exit threat room.",
                                      PLAYER_ORDER_TO_COLOR.at(player_order));
            this->log_watch_intervention(time_step, text);
        }

        void ASISTStudy3InterventionLogger::
            log_activate_ask_for_help_threat_intervention(int time_step,
                                                          int player_order,
                                                          int latency) {

            string text = fmt::format("{} did not ask for help to exit "
                                      "threat room in the last {} seconds.",
                                      PLAYER_ORDER_TO_COLOR.at(player_order),
                                      latency);
            this->log_activate_intervention(time_step, text);
        }

        void ASISTStudy3InterventionLogger::
            log_cancel_ask_for_help_threat_intervention(int time_step,
                                                        int player_order,
                                                        bool left_room,
                                                        bool help_requested,
                                                        bool being_released) {
            string text;

            if (left_room) {
                text = fmt::format("{} left room. ",
                                   PLAYER_ORDER_TO_COLOR.at(player_order));
            }
            if (help_requested) {
                text += fmt::format("{} asked for help. ",
                                    PLAYER_ORDER_TO_COLOR.at(player_order));
            }
            if (being_released) {
                text += fmt::format("{} is being released by the engineer.",
                                    PLAYER_ORDER_TO_COLOR.at(player_order));
            }

            text = fmt::format(
                "{}Canceling ask-for-help (threat) intervention.", text);

            this->log_cancel_intervention(time_step, text);
        }

        void ASISTStudy3InterventionLogger::
            log_hinder_ask_for_help_threat_intervention(int time_step,
                                                        int player_order) {
            string text = fmt::format(
                "{} needs help to exit a threat room but mentioned that "
                "recently. No need to watch for intervention.",
                PLAYER_ORDER_TO_COLOR.at(player_order));

            this->log(time_step, text);
        }

        void
        ASISTStudy3InterventionLogger::log_intervene_on_help_on_the_way(
            int time_step, int player_order) {

            string text =
                fmt::format("Help-on-the-way intervention for player {}.",
                            PLAYER_ORDER_TO_COLOR.at(player_order));
            this->log_trigger_intervention(time_step, text);
        }

        void ASISTStudy3InterventionLogger::
            log_watch_help_on_the_way_intervention(int time_step,
                                                       int player_order) {

            string text = fmt::format("{} asked for help.",
                                      PLAYER_ORDER_TO_COLOR.at(player_order));
            this->log_watch_intervention(time_step, text);
        }

        void ASISTStudy3InterventionLogger::
            log_activate_help_on_the_way_intervention(int time_step,
                                                          int player_order,
                                                          int latency) {

            string text = fmt::format("{} asked for help but no one "
                                      "replied in the last {} seconds.",
                                      PLAYER_ORDER_TO_COLOR.at(player_order),
                                      latency);
            this->log_activate_intervention(time_step, text);
        }

        void ASISTStudy3InterventionLogger::
            log_cancel_help_on_the_way_intervention(int time_step,
                                                        int assisted_player_order_,
                                                    int helper_player_order,
                                                        bool changed_area,
                                                        bool help_request_answered) {
            string text;

            if (changed_area) {
                text = fmt::format("{} changed area. ",
                                   PLAYER_ORDER_TO_COLOR.at(assisted_player_order_));
            }
            if (help_request_answered) {
                text += fmt::format("{} answered to {}'s help request. ",
                                    PLAYER_ORDER_TO_COLOR.at(helper_player_order),
                                    PLAYER_ORDER_TO_COLOR.at(assisted_player_order_));
            }

            text = fmt::format(
                "{}Canceling help-on-the-way intervention.", text);

            this->log_cancel_intervention(time_step, text);
        }

        void ASISTStudy3InterventionLogger::log_watch_intervention(
            int time_step, const string& text) {
            this->log_common_info(time_step_to_mission_timer(time_step),
                                  boost::trim_copy(text));
            this->log_file << "X\t";
            this->log_file << "\n";
            this->log_file.flush();
        }

        void ASISTStudy3InterventionLogger::log_cancel_intervention(
            int time_step, const string& text) {
            this->log_common_info(time_step_to_mission_timer(time_step),
                                  boost::trim_copy(text));
            this->log_file << "\t";  // watch
            this->log_file << "X\t"; // cancel
            this->log_file << "\n";
            this->log_file.flush();
        }

        void ASISTStudy3InterventionLogger::log_activate_intervention(
            int time_step, const string& text) {
            this->log_common_info(time_step_to_mission_timer(time_step),
                                  boost::trim_copy(text));
            this->log_file << "\t\t"; // watch and cancel
            this->log_file << "X";    // activate
            this->log_file << "\n";
            this->log_file.flush();
        }

        void ASISTStudy3InterventionLogger::log_trigger_intervention(
            int time_step, const string& text) {
            this->log_common_info(time_step_to_mission_timer(time_step),
                                  boost::trim_copy(text));
            this->log_file << "\t\t\t"; // watch, cancel, activate
            this->log_file << "X";      // intervene
            this->log_file << "\n";
            this->log_file.flush();
        }

        void ASISTStudy3InterventionLogger::log(int time_step,
                                                const string& text) {
            this->log_common_info(time_step_to_mission_timer(time_step),
                                  boost::trim_copy(text));
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
