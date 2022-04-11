#include "ASISTStudy3InterventionReporter.h"

#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <fmt/format.h>

#include "asist/study3/ASISTStudy3InterventionEstimator.h"
#include "utils/JSONChecker.h"

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
            json_data["duration"] = 10000;
            json_data["type"] = "string";
            json_data["renderers"] = nlohmann::json::array();
            json_data["renderers"].push_back("Minecraft_Chat");
            json_data["source"] = agent->get_id();
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

            // TODO - remove before merging with main branch
            intervention_message["topic"] =
                "agent/intervention/ASI_UAZ_TA1_ToMCAT/chat";

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

            check_field(this->json_settings, "missions_to_intervene");

            int mission_order =
                agent->get_evidence_metadata()[0]["mission_order"];
            unordered_set<int> missions_to_intervene = unordered_set<int>(
                this->json_settings["missions_to_intervene"]);

            if (!EXISTS(mission_order, missions_to_intervene)) {
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
                    this->intervene_on_ask_for_help(agent, t, messages);
                }
            }

            return messages;
        }

        void ASISTStudy3InterventionReporter::intervene_on_introduction(
            const AgentPtr& agent,
            int time_step,
            vector<nlohmann::json>& messages) {

            check_field(this->json_settings, "activations");
            check_field(this->json_settings, "introduction_time_step");
            check_field(this->json_settings["activations"], "introduction");

            if (!this->json_settings["activations"]["introduction"]) {
                this->introduced = true;
                return;
            }

            if (!this->introduced &&
                time_step >= this->json_settings["introduction_time_step"]) {
                this->introduced = true;

                messages.push_back(this->get_introductory_intervention_message(
                    agent, time_step));
                this->custom_logger->log_intervene_on_introduction(time_step);
            }
        }

        void ASISTStudy3InterventionReporter::intervene_on_motivation(
            const AgentPtr& agent,
            int time_step,
            vector<nlohmann::json>& messages) {

            check_field(this->json_settings, "activations");
            check_field(this->json_settings["activations"], "motivation");
            check_field(this->json_settings, "motivation_time_step");
            check_field(this->json_settings, "motivation_min_percentile");
            check_field(this->json_settings, "explanations");
            check_field(this->json_settings["explanations"], "motivation");

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
                int num_encouragements = estimator->get_num_encouragements();
                if (cdf <= min_percentile) {
                    auto motivation_msg =
                        this->get_motivation_intervention_message(agent,
                                                                  time_step);
                    motivation_msg["data"]["explanation"] = fmt::format(
                        (string)this
                            ->json_settings["explanations"]["motivation"],
                        cdf);
                    messages.push_back(motivation_msg);

                    this->custom_logger->log_intervene_on_motivation(
                        time_step, num_encouragements, cdf);
                }
                else {
                    this->custom_logger->log_cancel_motivation_intervention(
                        time_step, num_encouragements, cdf);
                }
            }
        }

        void ASISTStudy3InterventionReporter::intervene_on_communication_marker(
            const AgentPtr& agent,
            int time_step,
            vector<nlohmann::json>& messages) {

            check_field(this->json_settings, "activations");
            check_field(this->json_settings["activations"],
                        "communication_marker");

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
                    this->custom_logger->log_intervene_on_marker(time_step,
                                                                 player_order);
                }
            }
        }

        void ASISTStudy3InterventionReporter::intervene_on_ask_for_help(
            const AgentPtr& agent,
            int time_step,
            vector<nlohmann::json>& messages) {

            check_field(this->json_settings, "activations");
            check_field(this->json_settings["activations"], "ask_for_help");

            if (!this->json_settings["activations"]["ask_for_help"]) {
                return;
            }

            auto estimator =
                dynamic_pointer_cast<ASISTStudy3InterventionEstimator>(
                    agent->get_estimators()[0]);

            const auto& critical_victim =
                estimator->get_active_no_critical_victim_help_request();
            const auto& threat = estimator->get_active_no_threat_help_request();
            for (int player_order = 0; player_order < 3; player_order++) {
                if (critical_victim.at(player_order)) {
                    auto intervention_msg =
                        this->get_ask_for_help_intervention_message(
                            agent, time_step, player_order);

                    messages.push_back(intervention_msg);

                    estimator->clear_active_ask_for_help_critical_victim(
                        player_order);
                    this->custom_logger
                        ->log_intervene_on_ask_for_help_critical_victim(
                            time_step, player_order);
                }
                if (threat.at(player_order)) {
                    auto intervention_msg =
                        this->get_ask_for_help_intervention_message(
                            agent, time_step, player_order);

                    messages.push_back(intervention_msg);

                    estimator->clear_active_ask_for_help_threat(player_order);
                    this->custom_logger->log_intervene_on_ask_for_help_threat(
                        time_step, player_order);
                }
            }
        }

        nlohmann::json
        ASISTStudy3InterventionReporter::get_introductory_intervention_message(
            const AgentPtr& agent, int time_step) const {

            check_field(this->json_settings, "prompts");
            check_field(this->json_settings["prompts"], "introduction");
            check_field(this->json_settings, "explanations");
            check_field(this->json_settings["explanations"], "introduction");

            nlohmann::json intervention_message =
                this->get_template_intervention_message(agent, time_step);
            intervention_message["data"]["content"] =
                this->json_settings["prompts"]["introduction"];
            intervention_message["data"]["explanation"]["info"] =
                this->json_settings["explanations"]["introduction"];
            intervention_message["data"]["receivers"] = this->player_ids;

            return intervention_message;
        }

        nlohmann::json
        ASISTStudy3InterventionReporter::get_motivation_intervention_message(
            const AgentPtr& agent, int time_step) const {

            check_field(this->json_settings, "prompts");
            check_field(this->json_settings["prompts"], "motivation");

            nlohmann::json intervention_message =
                this->get_template_intervention_message(agent, time_step);
            intervention_message["data"]["content"] =
                this->json_settings["prompts"]["motivation"];
            intervention_message["data"]["receivers"] = this->player_ids;

            return intervention_message;
        }

        nlohmann::json ASISTStudy3InterventionReporter::
            get_communication_marker_intervention_message(
                const AgentPtr& agent,
                int time_step,
                int player_order,
                const ASISTStudy3MessageConverter::Marker& marker) const {

            check_field(this->json_settings, "prompts");
            check_field(this->json_settings["prompts"], "communication_marker");
            check_field(this->json_settings, "explanations");
            check_field(this->json_settings["explanations"],
                        "communication_marker");

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
            intervention_message["data"]["receivers"] = nlohmann::json::array();
            intervention_message["data"]["receivers"].push_back(player_id);
            const string& explanation =
                this->json_settings["explanations"]["ask_for_help"];
            intervention_message["data"]["explanation"] = explanation;

            return intervention_message;
        }

        nlohmann::json
        ASISTStudy3InterventionReporter::get_ask_for_help_intervention_message(
            const AgentPtr& agent, int time_step, int player_order) const {

            check_field(this->json_settings, "prompts");
            check_field(this->json_settings["prompts"], "ask_for_help");
            check_field(this->json_settings, "explanations");
            check_field(this->json_settings["explanations"], "ask_for_help");

            nlohmann::json intervention_message =
                this->get_template_intervention_message(agent, time_step);
            const string& prompt =
                this->json_settings["prompts"]["ask_for_help"];
            string player_color = player_order_to_color(player_order);

            string player_id = player_ids_per_color[player_order];
            intervention_message["data"]["content"] =
                fmt::format(prompt, player_color);
            intervention_message["data"]["receivers"] = nlohmann::json::array();
            intervention_message["data"]["receivers"].push_back(player_id);
            const string& explanation =
                this->json_settings["explanations"]["ask_for_help"];
            intervention_message["data"]["explanation"] = fmt::format(
                explanation,
                ASISTStudy3InterventionEstimator::ASK_FOR_HELP_LATENCY);

            return intervention_message;
        }

        void ASISTStudy3InterventionReporter::prepare() {
            this->player_info_initialized = false;
            this->introduced = false;
            this->intervened_on_motivation = false;
        }

        void ASISTStudy3InterventionReporter::set_logger(
            const OnlineLoggerPtr& logger) {
            ASISTReporter::set_logger(logger);
            if (const auto& tmp =
                    dynamic_pointer_cast<ASISTStudy3InterventionLogger>(
                        logger)) {
                // We store a reference to the logger into a local variable to
                // avoid casting throughout the code.
                this->custom_logger = tmp;
            }
            else {
                throw TomcatModelException(
                    "The ASISTStudy3InterventionEstimator requires a "
                    "logger of type ASISTStudy3InterventionLogger.");
            }
        }

    } // namespace model
} // namespace tomcat
