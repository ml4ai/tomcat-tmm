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
            // equals to 1 (one trial being processed). However, in the
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
            intervention_message["data"]["explanation"]["info"] =
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
