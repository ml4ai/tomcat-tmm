#include "ASISTStudy3InterventionReporter.h"

#include <boost/algorithm/string/join.hpp>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <fmt/format.h>

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        ASISTStudy3InterventionReporter::ASISTStudy3InterventionReporter() {}

        ASISTStudy3InterventionReporter::~ASISTStudy3InterventionReporter() {}

        //----------------------------------------------------------------------
        // Copy & Move constructors/assignments
        //----------------------------------------------------------------------
        ASISTStudy3InterventionReporter::ASISTStudy3InterventionReporter(
            const ASISTStudy3InterventionReporter& reporter) {}

        ASISTStudy3InterventionReporter&
        ASISTStudy3InterventionReporter::operator=(
            const ASISTStudy3InterventionReporter& reporter) {
            return *this;
        }

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        vector<nlohmann::json>
        ASISTStudy3InterventionReporter::translate_estimates_to_messages(
            const AgentPtr& agent, int time_step) {
            vector<nlohmann::json> messages;

            // Interventions after 2.5, 5 and 8 minutes since the mission start
            if (EXISTS(time_step, this->all_intervention_times)) {
                nlohmann::json intervention_message;
                intervention_message["header"] =
                    this->get_header_section(agent, "agent");
                intervention_message["msg"] =
                    this->get_msg_section(agent, "Intervention:Chat");
                intervention_message["data"] =
                    this->get_common_data_section(agent, time_step);

                if (time_step == this->intro_time) {
                    // Introduce agent
                    intervention_message["data"]["content"] =
                        this->get_introductory_speech();
                    intervention_message["data"]["explanation"]["text"] =
                        "ToMCAT introduces itself to the team.";
                    messages.push_back(intervention_message);
                }

                if (time_step == this->closing_time) {
                    // Say goodbye
                    intervention_message["data"]["content"] =
                        this->get_closing_speech();
                    intervention_message["data"]["explanation"]["text"] =
                        "ToMCAT says goodbye to the team letting them know how "
                        "well they performed compared to other teams.";
                    messages.push_back(intervention_message);
                }

                if (EXISTS(time_step, this->mission_timer_alert_times)) {
                    // Say goodbye
                    intervention_message["data"]["content"] =
                        this->get_mission_timer_alert_speech(time_step);
                    intervention_message["data"]["explanation"]["text"] =
                        "It warns the team about the remaining time of the "
                        "mission halfway through it and 2 minutes before it "
                        "ends.";
                    messages.push_back(intervention_message);
                }

                for (const auto& estimator : agent->get_estimators()) {
                    for (const auto& base_estimator :
                         estimator->get_base_estimators()) {

                        if (base_estimator->get_estimates().label ==
                            "TeamQuality") {
                            if (EXISTS(time_step,
                                       this->performance_feedback_times)) {

                                auto [current_quality, confidence] =
                                    this->get_estimated_team_quality(
                                        base_estimator, time_step);
                                intervention_message["data"]["content"] =
                                    this->get_team_quality_speech(
                                        current_quality, confidence);
                                intervention_message["data"]["duration"] = 90;
                                intervention_message
                                    ["data"]["explanation"]["text"] =
                                        "Gives feedback to the team based on "
                                        "their "
                                        "quality. The feedback is inferred "
                                        "from the current "
                                        "team score, mission timer and amount "
                                        "of seconds each player "
                                        "has spent do far in each one of the 6 "
                                        "sections of the map.";
                                this->last_quality = current_quality;
                                messages.push_back(intervention_message);
                            }
                        }
                        else if (base_estimator->get_estimates().label ==
                                 "TeamQualityDecay") {
                            if (EXISTS(time_step,
                                       this->map_section_check_times)) {

                                int player_number = 0;
                                const auto& json_players =
                                    agent->get_evidence_metadata()["players"];
                                for (const auto json_player : json_players) {
                                    // This intervention is per player.
                                    intervention_message["data"]["receivers"]
                                        .clear();

                                    double quality_decay =
                                        this->get_team_quality_decay(
                                            base_estimator,
                                            time_step,
                                            player_number++);

                                    if (quality_decay > 0.5) {
                                        intervention_message
                                            ["data"]["receivers"]
                                                .push_back(json_player["id"]);
                                        intervention_message["data"]["conten"
                                                                     "t"] =
                                            this->get_team_quality_decay_speech();
                                        intervention_message["data"]
                                                            ["duration"] = 60;
                                        intervention_message
                                            ["data"]["explanation"]["text"] =
                                                "For each player, it infers "
                                                "whether staying in the "
                                                "current map section degrades "
                                                "the team quality by more than "
                                                "50%. If it does, it informs "
                                                "the player that it is better "
                                                "to explore a different area "
                                                "of the map.";
                                        messages.push_back(
                                            intervention_message);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            return messages;
        }

        string
        ASISTStudy3InterventionReporter::get_introductory_speech() const {
            return "Hi, Team. I am ToMCAT and I will assist you in this "
                   "mission. I will give you feedback about "
                   "the team performance, warn you about the remaining time "
                   "and give individual thoughts on whether  "
                   "keep exploring a certain section of the map (6 sections: 3 "
                   "evenly split sections on the top and bottom portions of "
                   "the map) is likely to degrade the team performance. Good "
                   "luck and team up!";
        }

        string ASISTStudy3InterventionReporter::get_closing_speech() const {
            string speech;

            if (this->last_quality == 0) {
                speech =
                    "ToMCAT, here. It was nice to assist you on this mission. "
                    "Compared to other teams, you did not do well in this "
                    "mission. Consider changing your strategy for the next "
                    "mission. See you next time!";
            }
            else if (this->last_quality == 1) {
                speech =
                    "ToMCAT, here. It was nice to assist you on this mission. "
                    "Compared to other teams, you had an average performance "
                    "in "
                    "this mission. Consider changing your strategy for the "
                    "next "
                    "mission. See you next time!";
            }
            else {
                speech =
                    "ToMCAT, here. It was nice to assist you on this mission. "
                    "Compared to other teams, you did very well in this "
                    "mission. See you next time!";
            }

            return speech;
        }

        string ASISTStudy3InterventionReporter::get_mission_timer_alert_speech(
            int time_step) const {
            int minutes = (600 - time_step) / 60;

            string speech = fmt::format(
                "ToMCAT, here. Remember that you have only {} minutes until "
                "the end of the mission.",
                minutes);

            return speech;
        }

        string
        ASISTStudy3InterventionReporter::get_team_quality_decay_speech() const {
            string speech = fmt::format(
                "ToMCAT, here. Based on previous teams' strategies and your "
                "role, I suggest you explore a different section of the map to "
                "increase your team performance.");

            return speech;
        }

        nlohmann::json ASISTStudy3InterventionReporter::build_log_message(
            const AgentPtr& agent, const string& log) const {
            // No predefined format for the ASIST program
            nlohmann::json message;
            return message;
        }

        void ASISTStudy3InterventionReporter::prepare() {
            this->last_quality = -1;

            // 10 seconds after the mission starts
            this->intro_time = 10;

            // 10 before the mission ends
            this->closing_time = 590;

            // 5:00 min, 8:00 min
            this->mission_timer_alert_times = {300, 480};
            // 2:30 min, 5:10 min, 7:50 min, 9:40 min
            this->performance_feedback_times = {150, 310, 450, 580};

            // 2:00 min, 3:00 min, 4:00 min, 4:50, 6:00 min, 7:00 min, 8:00 min,
            // 9:00 min
            this->map_section_check_times = {
                120, 180, 240, 290, 360, 420, 470, 540};

            this->all_intervention_times.insert(this->intro_time);
            this->all_intervention_times.insert(this->closing_time);
            this->all_intervention_times.insert(
                this->mission_timer_alert_times.begin(),
                this->mission_timer_alert_times.end());
            this->all_intervention_times.insert(
                this->performance_feedback_times.begin(),
                this->performance_feedback_times.end());
            this->all_intervention_times.insert(
                this->map_section_check_times.begin(),
                this->map_section_check_times.end());
        }

        nlohmann::json ASISTStudy3InterventionReporter::get_header_section(
            const AgentPtr& agent, const string& message_type) const {
            nlohmann::json header;
            header["timestamp"] = this->get_current_timestamp();
            header["message_type"] = message_type;
            header["version"] = agent->get_version();

            return header;
        }

        nlohmann::json ASISTStudy3InterventionReporter::get_msg_section(
            const AgentPtr& agent, const string& sub_type) const {
            nlohmann::json msg_common;

            msg_common["trial_id"] =
                agent->get_evidence_metadata()["trial_unique_id"];
            msg_common["experiment_id"] =
                agent->get_evidence_metadata()["experiment_id"];
            msg_common["timestamp"] = this->get_current_timestamp();
            msg_common["source"] = agent->get_id();
            msg_common["version"] = "1.0";
            msg_common["trial_number"] =
                agent->get_evidence_metadata()["trial_id"];
            msg_common["sub_type"] = sub_type;

            return msg_common;
        }

        nlohmann::json ASISTStudy3InterventionReporter::get_common_data_section(
            const AgentPtr& agent, int time_step) {
            nlohmann::json msg_common;

            boost::uuids::uuid u = boost::uuids::random_generator()();
            msg_common["id"] = boost::uuids::to_string(u);
            msg_common["source"] = agent->get_id();
            msg_common["created"] = this->get_current_timestamp();
            msg_common["start"] = -1;
            msg_common["duration"] = 1;
            msg_common["receivers"] = this->get_player_list(agent);
            msg_common["type"] = "string";
            msg_common["renderers"] = nlohmann::json::array();
            msg_common["renderers"].push_back("Minecraft_Chat");

            return msg_common;
        }

        pair<int, double>
        ASISTStudy3InterventionReporter::get_estimated_team_quality(
            const EstimatorPtr& estimator, int time_step) const {
            double probability = -1;
            int quality = -1;
            for (int i = 0; i < estimator->get_estimates().estimates.size();
                 i++) {
                double temp =
                    estimator->get_estimates().estimates[i](0, time_step);
                if (temp > probability) {
                    probability = temp;
                    quality = i;
                }
            }

            return {quality, probability};
        }

        double ASISTStudy3InterventionReporter::get_team_quality_decay(
            const EstimatorPtr& estimator,
            int time_step,
            int player_number) const {
            return estimator->get_estimates().estimates[0](0, time_step);
        }

        string ASISTStudy3InterventionReporter::get_team_quality_speech(
            int quality, int confidence) const {

            string speech;
            if (quality == 0) {
                if (confidence > 0.5) {
                    speech = "Hi, Team. ToMCAT, here. Compared to other teams, "
                             "I am confident you are not doing well at this "
                             "point. Consider changing your strategy.";
                }
                else {
                    speech = "Hi, Team. ToMCAT, here. Compared to other teams, "
                             "I believe you are not doing well at this "
                             "point. Consider changing your strategy.";
                }
            }
            else if (quality == 1) {
                if (confidence > 0.5) {
                    speech =
                        "Hi, Team. ToMCAT, here. Compared to other teams, "
                        "I am confident you have an average performance at "
                        "this point. Consider changing your strategy.";
                }
                else {
                    speech = "Hi, Team. ToMCAT, here. Compared to other teams, "
                             "I believe you have an average performance at "
                             "this point. Consider changing your strategy.";
                }
            }
            else {
                if (confidence > 0.5) {
                    speech =
                        "Hi, Team. ToMCAT, here. Compared to other teams, "
                        "I am confident you are doing very well at this point.";
                }
                else {
                    speech = "Hi, Team. ToMCAT, here. Compared to other teams, "
                             "I believe you are doing very well at this point.";
                }
            }

            return speech;
        }

        nlohmann::json ASISTStudy3InterventionReporter::get_player_list(
            const AgentPtr& agent) const {
            nlohmann::json player_ids = nlohmann::json::array();

            for (const auto json_player :
                 agent->get_evidence_metadata()["players"]) {
                player_ids.push_back(json_player["id"]);
            }

            return player_ids;
        }

        string
        ASISTStudy3InterventionReporter::get_timestamp_at(const AgentPtr& agent,
                                                          int time_step) const {
            const string& initial_timestamp =
                agent->get_evidence_metadata()["initial_timestamp"];
            int elapsed_time =
                time_step * (int)agent->get_evidence_metadata()["step_size"];

            return this->get_elapsed_timestamp(initial_timestamp, elapsed_time);
        }

        int ASISTStudy3InterventionReporter::get_milliseconds_at(
            const AgentPtr& agent, int time_step) const {
            int step_size = agent->get_evidence_metadata()["step_size"];
            return time_step * step_size * 1000;
        }
    } // namespace model
} // namespace tomcat
