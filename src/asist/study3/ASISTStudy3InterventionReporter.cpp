#include "ASISTStudy3InterventionReporter.h"

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
        ASISTStudy3InterventionReporter::ASISTStudy3InterventionReporter(
            const nlohmann::json& settings)
            : settings(settings), introduced(false) {}

        //----------------------------------------------------------------------
        // Copy & Move constructors/assignments
        //----------------------------------------------------------------------
        ASISTStudy3InterventionReporter::ASISTStudy3InterventionReporter(
            const ASISTStudy3InterventionReporter& reporter) {
            this->copy(reporter);
        }

        ASISTStudy3InterventionReporter&
        ASISTStudy3InterventionReporter::operator=(
            const ASISTStudy3InterventionReporter& reporter) {
            this->copy(reporter);
            return *this;
        }

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        void ASISTStudy3InterventionReporter::copy(
            const ASISTStudy3InterventionReporter& reporter) {
            this->settings = reporter.settings;
            this->introduced = false;
        }

        vector<nlohmann::json>
        ASISTStudy3InterventionReporter::translate_estimates_to_messages(
            const AgentPtr& agent, int time_step) {
            vector<nlohmann::json> messages;

            if (this->introduced) {
                // The agent only checks for interventions after it has
                // introduced himself to the team.
            }
            else {
                if (time_step >= this->settings["introduction_time_step"]) {
                    this->introduced = true;
                    messages.push_back(
                        this->get_introductory_intervention_message(agent,
                                                                    time_step));
                }
            }

            //
            //            // Interventions after 2.5, 5 and 8 minutes since the
            //            mission start if (EXISTS(time_step,
            //            this->all_intervention_times)) {
            //                nlohmann::json intervention_message;
            //                intervention_message["header"] =
            //                    this->get_header_section(agent, "agent");
            //                intervention_message["msg"] =
            //                    this->get_msg_section(agent,
            //                    "Intervention:Chat");
            //                intervention_message["data"] =
            //                    this->get_common_data_section(agent,
            //                    time_step);
            //
            //                if (time_step == this->intro_time) {
            //                    // Introduce agent
            //                    intervention_message["data"]["content"] =
            //                        this->get_introductory_speech();
            //                    intervention_message["data"]["explanation"]["text"]
            //                    =
            //                        "ToMCAT introduces itself to the team.";
            //                    messages.push_back(intervention_message);
            //                }
            //
            //                if (time_step == this->closing_time) {
            //                    // Say goodbye
            //                    intervention_message["data"]["content"] =
            //                        this->get_closing_speech();
            //                    intervention_message["data"]["explanation"]["text"]
            //                    =
            //                        "ToMCAT says goodbye to the team letting
            //                        them know how " "well they performed
            //                        compared to other teams.";
            //                    messages.push_back(intervention_message);
            //                }
            //
            //                if (EXISTS(time_step,
            //                this->mission_timer_alert_times)) {
            //                    // Say goodbye
            //                    intervention_message["data"]["content"] =
            //                        this->get_mission_timer_alert_speech(time_step);
            //                    intervention_message["data"]["explanation"]["text"]
            //                    =
            //                        "It warns the team about the remaining
            //                        time of the " "mission halfway through it
            //                        and 2 minutes before it " "ends.";
            //                    messages.push_back(intervention_message);
            //                }
            //
            //                for (const auto& estimator :
            //                agent->get_estimators()) {
            //                    for (const auto& base_estimator :
            //                         estimator->get_base_estimators()) {
            //
            //                        if (base_estimator->get_estimates().label
            //                        ==
            //                            "TeamQuality") {
            //                            if (EXISTS(time_step,
            //                                       this->performance_feedback_times))
            //                                       {
            //
            //                                auto [current_quality, confidence]
            //                                =
            //                                    this->get_estimated_team_quality(
            //                                        base_estimator,
            //                                        time_step);
            //                                intervention_message["data"]["content"]
            //                                =
            //                                    this->get_team_quality_speech(
            //                                        current_quality,
            //                                        confidence);
            //                                intervention_message["data"]["duration"]
            //                                = 90; intervention_message
            //                                    ["data"]["explanation"]["text"]
            //                                    =
            //                                        "Gives feedback to the
            //                                        team based on " "their "
            //                                        "quality. The feedback is
            //                                        inferred " "from the
            //                                        current " "team score,
            //                                        mission timer and amount "
            //                                        "of seconds each player "
            //                                        "has spent do far in each
            //                                        one of the 6 " "sections
            //                                        of the map.";
            //                                this->last_quality =
            //                                current_quality;
            //                                messages.push_back(intervention_message);
            //                            }
            //                        }
            //                        else if
            //                        (base_estimator->get_estimates().label ==
            //                                 "TeamQualityDecay") {
            //                            if (EXISTS(time_step,
            //                                       this->map_section_check_times))
            //                                       {
            //
            //                                int player_number = 0;
            //                                const auto& json_players =
            //                                    agent->get_evidence_metadata()["players"];
            //                                for (const auto json_player :
            //                                json_players) {
            //                                    // This intervention is per
            //                                    player.
            //                                    intervention_message["data"]["receivers"]
            //                                        .clear();
            //
            //                                    double quality_decay =
            //                                        this->get_team_quality_decay(
            //                                            base_estimator,
            //                                            time_step,
            //                                            player_number++);
            //
            //                                    if (quality_decay > 0.5) {
            //                                        intervention_message
            //                                            ["data"]["receivers"]
            //                                                .push_back(json_player["id"]);
            //                                        intervention_message["data"]["conten"
            //                                                                     "t"] =
            //                                            this->get_team_quality_decay_speech();
            //                                        intervention_message["data"]
            //                                                            ["duration"]
            //                                                            = 60;
            //                                        intervention_message
            //                                            ["data"]["explanation"]["text"]
            //                                            =
            //                                                "For each player,
            //                                                it infers "
            //                                                "whether staying
            //                                                in the " "current
            //                                                map section
            //                                                degrades " "the
            //                                                team quality by
            //                                                more than " "50%.
            //                                                If it does, it
            //                                                informs " "the
            //                                                player that it is
            //                                                better " "to
            //                                                explore a
            //                                                different area "
            //                                                "of the map.";
            //                                        messages.push_back(
            //                                            intervention_message);
            //                                    }
            //                                }
            //                            }
            //                        }
            //                    }
            //                }
            //            }

            return messages;
        }

        nlohmann::json
        ASISTStudy3InterventionReporter::get_introductory_intervention_message(
            const AgentPtr& agent, int time_step) const {
            nlohmann::json intervention_message;
            this->add_header_section(
                intervention_message, agent, "agent", time_step);
            this->add_msg_section(
                intervention_message, agent, "Intervention:Chat", time_step);
            this->add_common_data_section(
                intervention_message, agent, time_step);
            intervention_message["data"]["content"] =
                this->settings["prompts"]["introduction"];

            return intervention_message;
        }

        nlohmann::json ASISTStudy3InterventionReporter::build_log_message(
            const AgentPtr& agent, const string& log) const {
            // No predefined format for the ASIST program
            nlohmann::json message;
            return message;
        }

        void ASISTStudy3InterventionReporter::prepare() {
            this->introduced = false;
        }

        void ASISTStudy3InterventionReporter::add_common_data_section(
            nlohmann::json& message,
            const AgentPtr& agent,
            int time_step) const {

            boost::uuids::uuid u = boost::uuids::random_generator()();
            message["id"] = boost::uuids::to_string(u);
            message["source"] = agent->get_id();
            message["created"] = this->get_timestamp_at(agent, time_step);
            message["start"] = -1;
            message["duration"] = 1;
            message["receivers"] = this->get_player_list(agent);
            message["type"] = "string";
            message["renderers"] = nlohmann::json::array();
            message["renderers"].push_back("Minecraft_Chat");
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

    } // namespace model
} // namespace tomcat
