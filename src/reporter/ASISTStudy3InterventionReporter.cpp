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

        nlohmann::json ASISTStudy3InterventionReporter::build_heartbeat_message(
            const AgentPtr& agent) {
            nlohmann::json msg;

            auto curr_time = chrono::steady_clock::now();
            std::chrono::duration<float> duration;
            duration = curr_time - this->time_last_heartbeat;
            if (duration.count() >= 10) {
                this->time_last_heartbeat = curr_time;
                // Creates a heartbeat message at every 10 seconds
                msg["header"] = this->get_header_section(agent, "status");
                msg["msg"] = this->get_msg_section(agent, "heartbeat");
                msg["data"]["state"] = "ok";
            }

            return msg;
        }

        nlohmann::json
        ASISTStudy3InterventionReporter::build_start_of_mission_message(
            const AgentPtr& agent) {
            return this->build_version_info_message(agent);
        }

        nlohmann::json
        ASISTStudy3InterventionReporter::build_end_of_mission_message(
            const AgentPtr& agent) {
            return this->build_version_info_message(agent);
        }

        nlohmann::json
        ASISTStudy3InterventionReporter::build_version_info_message(
            const AgentPtr& agent) {
            nlohmann::json msg;

            msg["header"] = this->get_header_section(agent, "agent");
            msg["msg"] = this->get_msg_section(agent, "versioninfo");
            msg["data"]["agent_name"] = agent->get_id();
            msg["data"]["version"] = agent->get_version();
            msg["data"]["owner"] = "UAZ";
            msg["data"]["config"] = nlohmann::json::array();
            nlohmann::json config;
            config["name"] = "model";
            config["value"] = "team_quality";
            msg["data"]["config"].push_back(config);
            msg["data"]["source"] = "https://gitlab.asist.aptima.com/asist/"
                                    "testbed/Agents/uaz_tmm_agent";
            msg["data"]["dependencies"] = nlohmann::json::array();
            msg["data"]["dependencies"].push_back("boost");
            msg["data"]["dependencies"].push_back("eigen");
            msg["data"]["dependencies"].push_back("gsl");
            msg["data"]["dependencies"].push_back("fmt");
            msg["data"]["dependencies"].push_back("nlohmann_json");
            msg["data"]["dependencies"].push_back("mosquitto");
            msg["data"]["publishes"] = nlohmann::json::array();
            nlohmann::json pub;
            // Topic 1
            pub["topic"] = "status/uaz_tmm_agent/heartbeats";
            pub["message_type"] = "status";
            pub["sub_type"] = "heartbeat";
            msg["data"]["publishes"].push_back(pub);
            // Topic 2
            pub["topic"] = "agent/uaz_tmm_agent/versioninfo";
            pub["message_type"] = "agent";
            pub["sub_type"] = "versioninfo";
            msg["data"]["publishes"].push_back(pub);
            // Topic 3
            pub["topic"] = "agent/intervention/uaz_tmm_agent/chat";
            pub["message_type"] = "agent";
            pub["sub_type"] = "Intervention:Chat";
            msg["data"]["publishes"].push_back(pub);
            // Topic 4
            pub["topic"] = "agent/control/rollcall/response";
            pub["message_type"] = "agent";
            pub["sub_type"] = "rollcall:response";
            msg["data"]["publishes"].push_back(pub);

            msg["data"]["subscribes"] = nlohmann::json::array();
            nlohmann::json sub;
            // Topic 1
            sub["topic"] = "trial";
            sub["message_type"] = "trial";
            sub["sub_type"] = "start";
            msg["data"]["subscribes"].push_back(sub);
            // Topic 2
            sub["topic"] = "observations/events/mission";
            sub["message_type"] = "event";
            sub["sub_type"] = "Event:MissionState";
            msg["data"]["subscribes"].push_back(sub);
            // Topic 3
            sub["topic"] = "observations/state";
            sub["message_type"] = "observation";
            sub["sub_type"] = "state";
            msg["data"]["subscribes"].push_back(sub);
            // Topic 4
            sub["topic"] = "observations/events/scoreboard";
            sub["message_type"] = "observation";
            sub["sub_type"] = "Event:Scoreboard";
            msg["data"]["subscribes"].push_back(sub);
            // Topic 5
            sub["topic"] = "agent/control/rollcall/request";
            sub["message_type"] = "agent";
            sub["sub_type"] = "rollcall:request";
            msg["data"]["subscribes"].push_back(sub);

            return msg;
        }

        nlohmann::json
        ASISTStudy3InterventionReporter::build_message_by_request(
            const AgentPtr& agent,
            const nlohmann::json& request_message,
            int time_step) {

            nlohmann::json response_msg;
            if (request_message["header"]["message_type"] == "agent" &&
                request_message["msg"]["sub_type"] == "rollcall:request") {
                response_msg = this->build_rollcall_message(
                    agent, request_message, time_step);
            }

            return response_msg;
        }

        nlohmann::json ASISTStudy3InterventionReporter::build_rollcall_message(
            const AgentPtr& agent,
            const nlohmann::json& request_message,
            int time_step) {

            nlohmann::json msg;
            msg["header"] = this->get_header_section(agent, "agent");
            msg["msg"] = this->get_msg_section(agent, "rollcall:response");
            msg["data"]["rollcall_id"] = request_message["data"]["rollcall_id"];
            msg["data"]["version"] = agent->get_version();
            msg["data"]["status"] = "up";
            int step_size = agent->get_evidence_metadata()["step_size"];
            msg["data"]["uptime"] = time_step * step_size;

            return msg;
        }

        string ASISTStudy3InterventionReporter::get_request_response_topic(
            const nlohmann::json& request_message,
            const MessageBrokerConfiguration& broker_config) {

            string topic;
            if (request_message["msg"]["sub_type"] == "rollcall:request") {
                topic = broker_config.rollcall_topic;
            }

            return topic;
        }

        vector<nlohmann::json>
        ASISTStudy3InterventionReporter::translate_estimates_to_messages(
            const AgentPtr& agent, int time_step) {
            vector<nlohmann::json> messages;

            // Interventions after 2.5, 5 and 8 minutes since the mission start
            if (time_step == 30 || time_step == 150 || time_step == 300 ||
                time_step == 480 || time_step == 570) {
                nlohmann::json intervention_message;
                intervention_message["header"] =
                    this->get_header_section(agent, "agent");
                intervention_message["msg"] =
                    this->get_msg_section(agent, "Intervention:Chat");
                intervention_message["data"] =
                    this->get_common_data_section(agent, time_step);

                if (time_step == 30) {
                    intervention_message["data"]["content"] =
                        this->get_introductory_speech();
                }
                else if (time_step == 570) {
                    intervention_message["data"]["content"] =
                        this->get_closing_speech();
                }
                else {
                    for (const auto& estimator : agent->get_estimators()) {
                        for (const auto& base_estimator :
                             estimator->get_base_estimators()) {
                            if (base_estimator->get_estimates().label ==
                                "TeamQuality") {

                                auto [quality, confidence] =
                                    this->get_estimated_team_quality(
                                        base_estimator, time_step);
                                intervention_message["data"]["content"] =
                                    this->get_tomcat_team_quality_speech(
                                        quality, confidence);
                                this->last_quality = quality;
                            }
                        }
                    }
                }

                messages.push_back(intervention_message);
            }

            return messages;
        }

        string
        ASISTStudy3InterventionReporter::get_introductory_speech() const {
            return "Hi, Team. I am ToMCAT and I will give you feedback about "
                   "your performance throughout the mission. Good luck and "
                   "team up!";
        }

        string ASISTStudy3InterventionReporter::get_closing_speech() const {
            string speech;

            if (this->last_quality == 0) {
                speech =
                    "ToMCAT, here. It was nice to assist you on this mission. "
                    "You did ok this time, but there's plenty of room for "
                    "improvement. See you next time!";
            }
            else if (this->last_quality == 1) {
                speech =
                    "ToMCAT, here. It was nice to assist you on this mission. "
                    "You did well this time, but there's still room for "
                    "improvement. See you next time!";
            }
            else {
                speech =
                    "ToMCAT, here. It was nice to assist you on this mission. "
                    "You did very well this time. See you next time!";
            }

            return speech;
        }

        nlohmann::json ASISTStudy3InterventionReporter::build_log_message(
            const AgentPtr& agent, const string& log) const {
            // No predefined format for the ASIST program
            nlohmann::json message;
            return message;
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

        string ASISTStudy3InterventionReporter::get_tomcat_team_quality_speech(
            int quality, int confidence) const {

            string speech;
            if (quality == 0) {
                if (confidence > 0.5) {
                    speech = "Hi, Team. ToMCAT, here. I am confident you are "
                             "not doing so well so far. Consider changing your "
                             "strategy.";
                }
                else {
                    speech = "Hi, Team. ToMCAT, here. I think you are not "
                             "doing so well so far. Consider changing your "
                             "strategy.";
                }
            }
            else if (quality == 1) {
                if (confidence > 0.5) {
                    speech = "Hi, Team. ToMCAT, here. I am confident you are "
                             "doing good, but you can do better.";
                }
                else {
                    speech = "Hi, Team. ToMCAT, here. I think you are "
                             "doing good, but you can do better.";
                }
            }
            else {
                if (confidence > 0.5) {
                    speech = "Hi, Team. ToMCAT, here. I am confident you are "
                             "doing an excellent job. Keep up the good work.";
                }
                else {
                    speech = "Hi, Team. ToMCAT, here. I think you are "
                             "doing an excellent job. Keep up the good work.";
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
