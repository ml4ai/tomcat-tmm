#include "ASISTStudy3InterventionReporter.h"

#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>
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

        nlohmann::json ASISTStudy3InterventionReporter::get_player_list(
            const AgentPtr& agent) {
            nlohmann::json player_ids = nlohmann::json::array();

            for (const auto& json_player :
                 agent->get_evidence_metadata()[0]["players"]) {
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
            const AgentPtr& agent, int time_step) {
            nlohmann::json intervention_message;
            add_header_section(
                intervention_message, agent, "agent", time_step, 0);
            add_msg_section(
                intervention_message, agent, "Intervention:Chat", time_step, 0);
            add_common_data_section(intervention_message, agent, time_step, 0);

            return intervention_message;
        }

        string ASISTStudy3InterventionReporter::player_order_to_color(
            int player_order) {

            if (player_order == 0) {
                return "Red";
            }
            else if (player_order == 1) {
                return "Green";
            }
            else if (player_order == 2) {
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
            if (!this->player_info_initialized) {
                const auto& json_trial = agent->get_evidence_metadata()[0];
                this->player_ids_per_color = vector<string>(3);
                for (const auto& json_player : json_trial["players"]) {
                    const string& player_color = json_player["color"];
                    if (boost::iequals(player_color, "red")) {
                        this->player_ids_per_color[0] = json_player["id"];
                    }
                    else if (boost::iequals(player_color, "green")) {
                        this->player_ids_per_color[1] = json_player["id"];
                    }
                    else if (boost::iequals(player_color, "blue")) {
                        this->player_ids_per_color[2] = json_player["id"];
                    }
                }
                this->player_ids = get_player_list(agent);
                this->player_info_initialized = true;
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

            this->store_player_info(agent);

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

                if (this->introduced) {
                    this->intervene_on_communication_marker(agent, t, messages);
                }
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

                messages.push_back(this->get_introductory_intervention_message(
                    agent, time_step));
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
                double cdf = estimator->get_encouragement_cdf();
                if (cdf <= min_percentile) {
                    auto motivation_msg =
                        this->get_motivation_intervention_message(agent,
                                                                  time_step);
                    motivation_msg["data"]["explanation"] = fmt::format(
                        (string)this
                            ->json_settings["explanations"]["motivation"],
                        cdf);
                    messages.push_back(motivation_msg);
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
            for (int player_order = 0; player_order < 3; player_order++) {
                const auto& marker = unspoken_markers.at(player_order);
                if (!marker.is_none()) {
                    auto intervention_msg =
                        this->get_communication_marker_intervention_message(
                            agent, time_step, player_order, marker);
                    messages.push_back(intervention_msg);

                    // The agent only intervenes once on each unspoken
                    // marker
                    estimator->clear_active_unspoken_marker(player_order);
                }
            }
        }

        nlohmann::json
        ASISTStudy3InterventionReporter::get_introductory_intervention_message(
            const AgentPtr& agent, int time_step) const {
            nlohmann::json intervention_message =
                this->get_template_intervention_message(agent, time_step);
            intervention_message["data"]["content"] =
                this->json_settings["prompts"]["introduction"];
            intervention_message["data"]["explanation"]["info"] =
                this->json_settings["explanations"]["introduction"];
            intervention_message["receivers"] = this->player_ids;

            return intervention_message;
        }

        nlohmann::json
        ASISTStudy3InterventionReporter::get_motivation_intervention_message(
            const AgentPtr& agent, int time_step) const {
            nlohmann::json intervention_message =
                this->get_template_intervention_message(agent, time_step);
            intervention_message["data"]["content"] =
                this->json_settings["prompts"]["motivation"];
            intervention_message["receivers"] = this->player_ids;

            return intervention_message;
        }

        nlohmann::json ASISTStudy3InterventionReporter::
            get_communication_marker_intervention_message(
                const AgentPtr& agent,
                int time_step,
                int player_order,
                const ASISTStudy3MessageConverter::Marker& marker) const {

            nlohmann::json intervention_message =
                this->get_template_intervention_message(agent, time_step);
            const string& prompt =
                this->json_settings["prompts"]["communication_marker"];
            string player_color = player_order_to_color(player_order);
            string marker_type =
                ASISTStudy3MessageConverter::MARKER_TYPE_TO_TEXT.at(
                    marker.type);

            string player_id = player_ids_per_color[player_order];
            intervention_message["data"]["content"] =
                fmt::format(prompt, player_color, marker_type);
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
            this->player_info_initialized = false;
            this->introduced = false;
            this->intervened_on_motivation = false;
        }

    } // namespace model
} // namespace tomcat
