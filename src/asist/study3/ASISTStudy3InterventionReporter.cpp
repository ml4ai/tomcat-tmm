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
            message["data"] = json_data;
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
                    this->intervene_on_marker(agent, t, messages);
                    this->intervene_on_help_request(agent, t, messages);
                    this->intervene_on_help_request_reply(agent, t, messages);
                }
            }

            return messages;
        }

        nlohmann::json
        ASISTStudy3InterventionReporter::get_template_intervention_message(
            const AgentPtr& agent,
            int time_step,
            const nlohmann::json& receivers,
            const string& intervention_type) const {

            check_field(this->json_settings, "prompts");
            check_field(this->json_settings["prompts"], intervention_type);
            check_field(this->json_settings, "explanations");
            check_field(this->json_settings["explanations"], intervention_type);

            nlohmann::json intervention_message;
            add_header_section(
                intervention_message, agent, "agent", time_step, 0);
            add_msg_section(
                intervention_message, agent, "Intervention:Chat", time_step, 0);
            add_common_data_section(intervention_message, agent, time_step, 0);

            intervention_message["data"]["content"] =
                this->json_settings["prompts"][intervention_type];
            intervention_message["data"]["receivers"] = receivers;
            intervention_message["data"]["explanation"]["info"] =
                this->json_settings["explanations"][intervention_type];

            return intervention_message;
        }

        void ASISTStudy3InterventionReporter::intervene_on_introduction(
            const AgentPtr& agent,
            int time_step,
            vector<nlohmann::json>& messages) {

            check_field(this->json_settings, "activations");
            check_field(this->json_settings, "introduction_time_step");
            check_field(this->json_settings["activations"], "introduction");

            int mission_order =
                agent->get_evidence_metadata()[0]["mission_order"];

            if (!this->json_settings["activations"]["introduction"] ||
                mission_order > 1) {
                // The agent only introduces itself in the first mission
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
                        this->get_motivation_intervention_message(
                            agent, time_step, cdf);
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

        void ASISTStudy3InterventionReporter::intervene_on_marker(
            const AgentPtr& agent,
            int time_step,
            vector<nlohmann::json>& messages) {

            check_field(this->json_settings, "activations");
            check_field(this->json_settings["activations"], "marker_block");

            if (!this->json_settings["activations"]["marker_block"]) {
                return;
            }

            auto estimator =
                dynamic_pointer_cast<ASISTStudy3InterventionEstimator>(
                    agent->get_estimators()[0]);

            for (int player_order = 0; player_order < 3; player_order++) {
                if (estimator->is_marker_intervention_active(player_order)) {
                    const auto& marker =
                        estimator->get_active_marker(player_order);
                    auto intervention_msg =
                        this->get_marker_intervention_message(
                            agent, time_step, player_order, marker);
                    messages.push_back(intervention_msg);

                    // The agent only intervenes once on each unspoken
                    // marker
                    estimator->restart_marker_intervention(player_order);
                    this->custom_logger->log_intervene_on_marker(time_step,
                                                                 player_order);
                }
            }
        }

        void ASISTStudy3InterventionReporter::intervene_on_help_request(
            const AgentPtr& agent,
            int time_step,
            vector<nlohmann::json>& messages) {

            check_field(this->json_settings, "activations");
            check_field(this->json_settings["activations"], "help_request");

            if (!this->json_settings["activations"]["help_request"]) {
                return;
            }

            auto estimator =
                dynamic_pointer_cast<ASISTStudy3InterventionEstimator>(
                    agent->get_estimators()[0]);

            for (int player_order = 0; player_order < 3; player_order++) {
                if (estimator
                        ->is_help_request_critical_victim_intervention_active(
                            player_order)) {
                    auto intervention_msg =
                        this->get_help_request_critical_victim_intervention_message(
                            agent, time_step, player_order);

                    messages.push_back(intervention_msg);

                    estimator
                        ->restart_help_request_critical_victim_intervention(
                            player_order);
                    this->custom_logger
                        ->log_intervene_help_request_critical_victim(
                            time_step, player_order);
                }
                if (estimator->is_help_request_room_escape_intervention_active(
                        player_order)) {
                    auto intervention_msg =
                        this->get_help_request_room_escape_intervention_message(
                            agent, time_step, player_order);

                    messages.push_back(intervention_msg);

                    estimator->restart_help_request_room_escape_intervention(
                        player_order);
                    this->custom_logger
                        ->log_intervene_on_help_request_room_escape(
                            time_step, player_order);
                }
            }
        }

        void ASISTStudy3InterventionReporter::intervene_on_help_request_reply(
            const AgentPtr& agent,
            int time_step,
            vector<nlohmann::json>& messages) {

            check_field(this->json_settings, "activations");
            check_field(this->json_settings["activations"],
                        "help_request_reply");

            if (!this->json_settings["activations"]["help_request_reply"]) {
                return;
            }

            auto estimator =
                dynamic_pointer_cast<ASISTStudy3InterventionEstimator>(
                    agent->get_estimators()[0]);

            for (int player_order = 0; player_order < 3; player_order++) {
                if (estimator->is_help_request_reply_intervention_active(
                        player_order)) {
                    auto intervention_msg =
                        this->get_help_request_reply_intervention_message(
                            agent, time_step, player_order);

                    messages.push_back(intervention_msg);

                    estimator->restart_help_request_reply_intervention(
                        player_order);
                    this->custom_logger->log_intervene_on_help_request_reply(
                        time_step, player_order);
                }
            }
        }

        nlohmann::json
        ASISTStudy3InterventionReporter::get_introductory_intervention_message(
            const AgentPtr& agent, int time_step) const {

            nlohmann::json intervention_message =
                this->get_template_intervention_message(
                    agent, time_step, this->player_ids, "introduction");

            return intervention_message;
        }

        nlohmann::json
        ASISTStudy3InterventionReporter::get_motivation_intervention_message(
            const AgentPtr& agent, int time_step, double cdf) const {

            nlohmann::json intervention_message =
                this->get_template_intervention_message(
                    agent, time_step, this->player_ids, "motivation");
            intervention_message["data"]["explanation"]["info"] = fmt::format(
                (string)intervention_message["data"]["explanation"]["info"],
                cdf);

            return intervention_message;
        }

        nlohmann::json
        ASISTStudy3InterventionReporter::get_marker_intervention_message(
            const AgentPtr& agent,
            int time_step,
            int player_order,
            const ASISTStudy3MessageConverter::Marker& marker) const {

            string player_color = player_order_to_color(player_order);
            nlohmann::json receivers = nlohmann::json::array();
            receivers.push_back(player_ids_per_color[player_order]);

            nlohmann::json intervention_message =
                this->get_template_intervention_message(
                    agent, time_step, receivers, "marker_block");

            string marker_type =
                ASISTStudy3MessageConverter::MARKER_TYPE_TO_TEXT.at(
                    marker.type);
            intervention_message["data"]["content"] =
                fmt::format((string)intervention_message["data"]["content"],
                            player_color,
                            marker_type);

            return intervention_message;
        }

        nlohmann::json ASISTStudy3InterventionReporter::
            get_help_request_critical_victim_intervention_message(
                const AgentPtr& agent, int time_step, int player_order) const {

            string player_color = player_order_to_color(player_order);
            nlohmann::json receivers = nlohmann::json::array();
            receivers.push_back(player_ids_per_color[player_order]);

            nlohmann::json intervention_message =
                this->get_template_intervention_message(
                    agent,
                    time_step,
                    receivers,
                    "help_request_critical_victim");

            intervention_message["data"]["content"] = fmt::format(
                (string)intervention_message["data"]["content"], player_color);
            intervention_message["data"]["explanation"]["info"] = fmt::format(
                (string)intervention_message["data"]["explanation"]["info"],
                ASISTStudy3InterventionEstimator::HELP_REQUEST_LATENCY);

            return intervention_message;
        }

        nlohmann::json ASISTStudy3InterventionReporter::
            get_help_request_room_escape_intervention_message(
                const AgentPtr& agent, int time_step, int player_order) const {

            string player_color = player_order_to_color(player_order);
            nlohmann::json receivers = nlohmann::json::array();
            receivers.push_back(player_ids_per_color[player_order]);

            nlohmann::json intervention_message =
                this->get_template_intervention_message(
                    agent, time_step, receivers, "help_request_room_escape");

            intervention_message["data"]["content"] = fmt::format(
                (string)intervention_message["data"]["content"], player_color);
            intervention_message["data"]["explanation"]["info"] = fmt::format(
                (string)intervention_message["data"]["explanation"]["info"],
                ASISTStudy3InterventionEstimator::HELP_REQUEST_LATENCY);

            return intervention_message;
        }

        nlohmann::json ASISTStudy3InterventionReporter::
            get_help_request_reply_intervention_message(
                const AgentPtr& agent, int time_step, int player_order) const {

            string player_color = player_order_to_color(player_order);
            nlohmann::json receivers = nlohmann::json::array();
            receivers.push_back(player_ids_per_color[player_order]);

            nlohmann::json intervention_message =
                this->get_template_intervention_message(
                    agent, time_step, receivers, "help_request_reply");

            intervention_message["data"]["content"] = fmt::format(
                (string)intervention_message["data"]["content"], player_color);
            intervention_message["data"]["explanation"]["info"] = fmt::format(
                (string)intervention_message["data"]["explanation"]["info"],
                ASISTStudy3InterventionEstimator::HELP_REQUEST_REPLY_LATENCY);

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
