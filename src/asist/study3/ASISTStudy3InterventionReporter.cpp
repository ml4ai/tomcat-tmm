#include "ASISTStudy3InterventionReporter.h"

#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <cctype>
#include <fmt/format.h>

#include "asist/study3/ASISTStudy3InterventionEstimator.h"

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        ASISTStudy3InterventionReporter::ASISTStudy3InterventionReporter(
            const nlohmann::json& json_settings)
            : ASISTReporter(json_settings) {}

        //----------------------------------------------------------------------
        // Copy & Move constructors/assignments
        //----------------------------------------------------------------------
        ASISTStudy3InterventionReporter::ASISTStudy3InterventionReporter(
            const ASISTStudy3InterventionReporter& reporter)
            : ASISTReporter(json_settings) {
            this->copy(reporter);
        }

        ASISTStudy3InterventionReporter&
        ASISTStudy3InterventionReporter::operator=(
            const ASISTStudy3InterventionReporter& reporter) {
            this->copy(reporter);
            return *this;
        }

        //----------------------------------------------------------------------
        // Static functions
        //----------------------------------------------------------------------

        nlohmann::json
        ASISTStudy3InterventionReporter::get_player_list(const AgentPtr& agent,
                                                         int data_point) {
            nlohmann::json player_ids = nlohmann::json::array();

            for (const auto& json_player :
                 agent->get_evidence_metadata()[data_point]["players"]) {
                player_ids.push_back(json_player["id"]);
            }

            return player_ids;
        }

        void ASISTStudy3InterventionReporter::add_common_data_section(
            nlohmann::json& message,
            const AgentPtr& agent,
            int time_step,
            int data_point) {

            nlohmann::json json_data;
            boost::uuids::uuid u = boost::uuids::random_generator()();
            json_data["id"] = boost::uuids::to_string(u);
            json_data["created"] =
                get_timestamp_at(agent, time_step, data_point);
            json_data["start"] = -1;
            json_data["duration"] = 1;
            json_data["type"] = "string";
            json_data["renderers"] = nlohmann::json::array();
            json_data["renderers"].push_back("Minecraft_Chat");
            message["data"] = json_data;
        }

        nlohmann::json
        ASISTStudy3InterventionReporter::get_template_intervention_message(
            const AgentPtr& agent, int time_step, int data_point) {
            nlohmann::json intervention_message;
            add_header_section(
                intervention_message, agent, "agent", time_step, data_point);
            add_msg_section(intervention_message,
                            agent,
                            "Intervention:Chat",
                            time_step,
                            data_point);
            add_common_data_section(
                intervention_message, agent, time_step, data_point);

            return intervention_message;
        }

        string ASISTStudy3InterventionReporter::player_order_to_color(
            int player_order) {

            if (player_order == 0) {
                return "Red";
            }
            else if (player_order == 0) {
                return "Green";
            }
            else if (player_order == 0) {
                return "Blue";
            }

            return "UNKNOWN";
        }

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        void ASISTStudy3InterventionReporter::copy(
            const ASISTStudy3InterventionReporter& reporter) {
            EstimateReporter::copy(reporter);
        }

        void ASISTStudy3InterventionReporter::store_player_info(
            const AgentPtr& agent) {
            int d = 0;
            for (const auto& json_trial : agent->get_evidence_metadata()) {
                vector<string> ids(3);
                for (const auto& json_player : json_trial["players"]) {
                    const string& player_color = json_player["color"];
                    if (boost::iequals(player_color, "red")) {
                        ids[0] = json_player["id"];
                    }
                    else if (boost::iequals(player_color, "green")) {
                        ids[1] = json_player["id"];
                    }
                    else if (boost::iequals(player_color, "blue")) {
                        ids[2] = json_player["id"];
                    }
                }
                this->player_ids_per_color.push_back(ids);
                this->player_ids.push_back(get_player_list(agent, d++));
            }
        }

        vector<nlohmann::json>
        ASISTStudy3InterventionReporter::translate_estimates_to_messages(
            const AgentPtr& agent, int time_step) {

            // In the online setting the number of data points will be
            // equals to 1 (1 trial being processed). However, in the
            // offline setting we might be processing multiple trials at the
            // same time, and we need to generate report messages for each
            // one of them.

            if (agent->get_evidence_metadata()[0]["mission_order"] == 1) {
                // We only intervene in mission 2
                return {};
            }

            vector<nlohmann::json> messages;
            auto estimator =
                dynamic_pointer_cast<ASISTStudy3InterventionEstimator>(
                    agent->get_estimators()[0]);

            int initial_time_step = time_step;
            int final_time_step = estimator->get_last_time_step();
            if (time_step == NO_OBS) {
                initial_time_step = 0;
            }

            for (int t = initial_time_step; t <= final_time_step; t++) {
                this->intervene_on_introduction(agent, t, messages);
                this->intervene_on_motivation(agent, t, messages);
            }

            //
            //            // Interventions after 2.5, 5 and 8 minutes since
            //            the mission start if (EXISTS(time_step,
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
            //                        "ToMCAT introduces itself to the
            //                        team.";
            //                    messages.push_back(intervention_message);
            //                }
            //
            //                if (time_step == this->closing_time) {
            //                    // Say goodbye
            //                    intervention_message["data"]["content"] =
            //                        this->get_closing_speech();
            //                    intervention_message["data"]["explanation"]["text"]
            //                    =
            //                        "ToMCAT says goodbye to the team
            //                        letting them know how " "well they
            //                        performed compared to other teams.";
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
            //                        time of the " "mission halfway through
            //                        it and 2 minutes before it " "ends.";
            //                    messages.push_back(intervention_message);
            //                }
            //
            //                for (const auto& estimator :
            //                agent->get_estimators()) {
            //                    for (const auto& base_estimator :
            //                         estimator->get_base_estimators()) {
            //
            //                        if
            //                        (base_estimator->get_estimates().label
            //                        ==
            //                            "TeamQuality") {
            //                            if (EXISTS(time_step,
            //                                       this->performance_feedback_times))
            //                                       {
            //
            //                                auto [current_quality,
            //                                confidence]
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
            //                                        team based on " "their
            //                                        " "quality. The
            //                                        feedback is inferred "
            //                                        "from the current "
            //                                        "team score, mission
            //                                        timer and amount " "of
            //                                        seconds each player "
            //                                        "has spent do far in
            //                                        each one of the 6 "
            //                                        "sections of the
            //                                        map.";
            //                                this->last_quality =
            //                                current_quality;
            //                                messages.push_back(intervention_message);
            //                            }
            //                        }
            //                        else if
            //                        (base_estimator->get_estimates().label
            //                        ==
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
            //                                    // This intervention is
            //                                    per player.
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
            //                                                                     "t"]
            //                                                                     =
            //                                            this->get_team_quality_decay_speech();
            //                                        intervention_message["data"]
            //                                                            ["duration"]
            //                                                            =
            //                                                            60;
            //                                        intervention_message
            //                                            ["data"]["explanation"]["text"]
            //                                            =
            //                                                "For each
            //                                                player, it
            //                                                infers "
            //                                                "whether
            //                                                staying in the
            //                                                " "current map
            //                                                section
            //                                                degrades "
            //                                                "the team
            //                                                quality by
            //                                                more than "
            //                                                "50%. If it
            //                                                does, it
            //                                                informs " "the
            //                                                player that it
            //                                                is better "
            //                                                "to explore a
            //                                                different area
            //                                                " "of the
            //                                                map.";
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

        void ASISTStudy3InterventionReporter::intervene_on_introduction(
            const AgentPtr& agent,
            int time_step,
            vector<nlohmann::json>& messages) {

            if (!this->json_settings["activations"]["introduction"]) {
                return;
            }

            if (!this->introduced &&
                time_step >= this->json_settings["introduction_time_step"]) {
                this->introduced = true;

                for (int d = 0; d < agent->get_evidence_metadata().size();
                     d++) {
                    messages.push_back(
                        this->get_introductory_intervention_message(
                            agent, time_step, d));
                }
            }
        }

        void ASISTStudy3InterventionReporter::intervene_on_motivation(
            const AgentPtr& agent,
            int time_step,
            vector<nlohmann::json>& messages) {

            if (!this->json_settings["activations"]["motivation"]) {
                return;
            }

            if (!this->intervened_on_motivation &&
                time_step >= this->json_settings["motivation_time_step"]) {
                this->intervened_on_motivation = true;

                double min_percentile =
                    (double)this->json_settings["motivation_min_percentile"] /
                    100;

                auto estimator =
                    dynamic_pointer_cast<ASISTStudy3InterventionEstimator>(
                        agent->get_estimators()[0]);
                const auto& cdfs = estimator->get_encouragement_cdfs();
                for (int d = 0; d < agent->get_evidence_metadata().size();
                     d++) {
                    if (cdfs(d) <= min_percentile) {
                        auto motivation_msg =
                            this->get_motivation_intervention_message(
                                agent, time_step, d);
                        motivation_msg["data"]["explanation"] = fmt::format(
                            (string)this
                                ->json_settings["explanations"]["motivation"],
                            cdfs(d));
                        messages.push_back(motivation_msg);
                    }
                }
            }
        }

        void ASISTStudy3InterventionReporter::intervene_on_communication_marker(
            const AgentPtr& agent,
            int time_step,
            vector<nlohmann::json>& messages) {

            if (!this->json_settings["activations"]["communication_marker"]) {
                return;
            }

            auto estimator =
                dynamic_pointer_cast<ASISTStudy3InterventionEstimator>(
                    agent->get_estimators()[0]);

            const auto& unspoken_markers =
                estimator->get_active_unspoken_markers();
            for (int d = 0; d < agent->get_evidence_metadata().size(); d++) {
                for (int player_order = 0;
                     player_order < unspoken_markers[d].size();
                     player_order++) {
                    auto intervention_msg =
                        this->get_communication_marker_intervention_message(
                            agent, time_step, d, player_order);
                    messages.push_back(intervention_msg);

                    // The agent only intervenes once on each unspoken marker
                    estimator->clear_active_unspoken_marker(player_order, d);
                }
            }
        }

        nlohmann::json
        ASISTStudy3InterventionReporter::get_introductory_intervention_message(
            const AgentPtr& agent, int time_step, int data_point) const {
            nlohmann::json intervention_message =
                this->get_template_intervention_message(
                    agent, time_step, data_point);
            intervention_message["data"]["content"] =
                this->json_settings["prompts"]["introduction"];
            intervention_message["data"]["explanation"] =
                this->json_settings["explanations"]["introduction"];
            intervention_message["receivers"] = this->player_ids[data_point];

            return intervention_message;
        }

        nlohmann::json
        ASISTStudy3InterventionReporter::get_motivation_intervention_message(
            const AgentPtr& agent, int time_step, int data_point) const {
            nlohmann::json intervention_message =
                this->get_template_intervention_message(
                    agent, time_step, data_point);
            intervention_message["data"]["content"] =
                this->json_settings["prompts"]["motivation"];
            intervention_message["receivers"] = this->player_ids[data_point];

            return intervention_message;
        }

        nlohmann::json ASISTStudy3InterventionReporter::
            get_communication_marker_intervention_message(
                const AgentPtr& agent,
                int time_step,
                int data_point,
                int player_order) const {
            nlohmann::json intervention_message =
                this->get_template_intervention_message(
                    agent, time_step, data_point);
            const string& prompt =
                this->json_settings["prompts"]["communication_intervention"];

            string player_color = player_order_to_color(player_order);
            string player_id = player_ids_per_color[data_point][player_order];
            intervention_message["data"]["content"] =
                fmt::format(prompt, player_color);
            intervention_message["receivers"] = nlohmann::json::array();
            intervention_message["receivers"].push_back(player_id);

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
            this->intervened_on_motivation = false;
        }

    } // namespace model
} // namespace tomcat
