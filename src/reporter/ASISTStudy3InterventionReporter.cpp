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
            if (time_step == 10 || time_step == 150 || time_step == 300 ||
                time_step == 480 || time_step == 570) {
                nlohmann::json intervention_message;
                intervention_message["header"] =
                    this->get_header_section(agent);
                intervention_message["msg"] = this->get_msg_section(agent);
                intervention_message["data"] =
                    this->get_common_data_section(agent, time_step);

                if (time_step == 10) {
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
            const AgentPtr& agent) const {
            nlohmann::json header;
            header["timestamp"] = this->get_current_timestamp();
            header["message_type"] = "agent";
            header["version"] = agent->get_version();

            return header;
        }

        nlohmann::json ASISTStudy3InterventionReporter::get_msg_section(
            const AgentPtr& agent) const {
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
            msg_common["sub_type"] = "Intervention:Chat";

            return msg_common;
        }

        nlohmann::json ASISTStudy3InterventionReporter::get_common_data_section(
            const AgentPtr& agent, int time_step) {
            nlohmann::json msg_common;

            boost::uuids::uuid u = boost::uuids::random_generator()();
            msg_common["id"] = boost::uuids::to_string(u);
            msg_common["agent"] = agent->get_id();
            msg_common["start"] = this->get_milliseconds_at(agent, time_step);
            msg_common["end"] =
                this->get_milliseconds_at(agent, time_step + 180);

            msg_common["receiver"] = this->get_player_list(agent);
            msg_common["type"] = "string";
            msg_common["renderer"] = "Minecraft_Chat";

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

        string ASISTStudy3InterventionReporter::get_player_list(
            const AgentPtr& agent) const {
            vector<string> player_ids;

            for (const auto json_player :
                 agent->get_evidence_metadata()["players"]) {
                player_ids.push_back(json_player["id"]);
            }

            return boost::algorithm::join(player_ids, ",");
        }

        string ASISTStudy3InterventionReporter::get_timestamp_at(
            const AgentPtr& agent, int time_step) const {
            const string& initial_timestamp =
                agent->get_evidence_metadata()["initial_timestamp"];
            int elapsed_time =
                time_step *
                (int)agent->get_evidence_metadata()["step_size"];

            return this->get_elapsed_timestamp(initial_timestamp, elapsed_time);
        }

        int ASISTStudy3InterventionReporter::get_milliseconds_at(
            const AgentPtr& agent, int time_step) const {
            int step_size = agent->get_evidence_metadata()["step_size"];
            return time_step * step_size * 1000;
        }
    } // namespace model
} // namespace tomcat
