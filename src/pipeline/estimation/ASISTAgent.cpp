#include "ASISTAgent.h"

#include <iomanip>
#include <time.h>

#include "utils/EigenExtensions.h"

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        ASISTAgent::ASISTAgent(const string& id,
                               const string& estimates_topic,
                               const string& log_topic,
                               const ASISTMessageConverter& message_converter)
            : Agent(id, estimates_topic, log_topic),
              message_converter(message_converter) {}

        ASISTAgent::~ASISTAgent() {}

        //----------------------------------------------------------------------
        // Copy & Move constructors/assignments
        //----------------------------------------------------------------------
        ASISTAgent::ASISTAgent(const ASISTAgent& agent)
            : Agent(agent.id, agent.estimates_topic, agent.log_topic),
              message_converter(agent.message_converter) {}

        ASISTAgent& ASISTAgent::operator=(const ASISTAgent& agent) {
            Agent::copy(agent);
            this->message_converter = agent.message_converter;
            return *this;
        }

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        EvidenceSet ASISTAgent::message_to_data(const nlohmann::json& message) {
            nlohmann::json log;
            return this->message_converter.get_data_from_message(message, log);
        }

        nlohmann::json ASISTAgent::estimates_to_message(
            const std::vector<std::shared_ptr<Estimator>>& estimators,
            int time_step) const {

            nlohmann::json messages = nlohmann::json::array();

            for (const auto& estimator : estimators) {
                for (const auto& base_estimator :
                     estimator->get_base_estimators()) {

                    nlohmann::json message;
                    message["TA"] = "TA1";
                    message["Team"] = "UAZ";
                    message["AgentID"] = this->id;
                    message["Trial"] =
                        this->message_converter.get_mission_trial_number();
                    message["TrainingCondition TriageSignal"] = "n.a";
                    message["TrainingCondition TriageNoSignal"] = "n.a";
                    message["TrainingCondition NoTriageNoSignal"] = "n.a";
                    message["VictimType Green"] = "n.a";
                    message["VictimType Yellow"] = "n.a";
                    message["VictimType Confidence"] = "n.a";

                    NodeEstimates estimates =
                        base_estimator->get_estimates_at(time_step);

                    int seconds_elapsed =
                        (time_step + base_estimator->get_inference_horizon()) *
                        this->message_converter.get_time_step_size();
                    time_t timestamp = this->message_converter
                                           .get_mission_initial_timestamp() +
                                       seconds_elapsed;
                    stringstream ss;
                    ss << put_time(localtime(&timestamp), "%Y-%m-%dT%T.000Z");
                    message["Timestamp"] = ss.str();

                    if (estimates.label == "TrainingCondition") {
                        message["TrainingCondition TriageSignal"] =
                            to_string(estimates.estimates.at(0));
                        message["TrainingCondition TriageNoSignal"] =
                            to_string(estimates.estimates.at(1));
                        message["TrainingCondition NoTriageNoSignal"] =
                            to_string(estimates.estimates.at(2));
                    }
                    else if (estimates.label == "Task") {
                        if (estimates.assignment(0) == 1) {
                            message["VictimType Green"] =
                                to_string(estimates.estimates.at(0));
                        }
                        else if (estimates.assignment(0) == 2) {
                            message["VictimType Yellow"] =
                                to_string(estimates.estimates.at(0));
                        }
                    }

                    messages.push_back(message);
                }
            }

            return messages;
        }

        unordered_set<string> ASISTAgent::get_topics_to_subscribe() const {
            return this->message_converter.get_used_topics();
        }

        nlohmann::json ASISTAgent::build_log_message(const string& log) const {
            nlohmann::json message;

            message["TA"] = "TA1";
            message["Team"] = "UAZ";
            message["AgentID"] = this->id;
            message["Trial"] =
                this->message_converter.get_mission_trial_number();
            message["log"] = log;

            return message;
        }

        void ASISTAgent::restart() {
            this->message_converter.start_new_mission();
        }

        bool ASISTAgent::is_mission_finished() const {
            return this->message_converter.is_mission_finished();
        }

    } // namespace model
} // namespace tomcat
