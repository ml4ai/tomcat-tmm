#include "ASISTAgent.h"

#include "utils/EigenExtensions.h"

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        ASISTAgent::ASISTAgent(const string& id,
                               const ASISTMessageConverter& message_converter)
            : Agent(id), message_converter(message_converter) {}

        ASISTAgent::~ASISTAgent() {}

        //----------------------------------------------------------------------
        // Copy & Move constructors/assignments
        //----------------------------------------------------------------------
        ASISTAgent::ASISTAgent(const ASISTAgent& agent)
            : Agent(agent.id), message_converter(agent.message_converter) {}

        ASISTAgent& ASISTAgent::operator=(const ASISTAgent& agent) {
            this->id = agent.id;
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

            size_t mission_init =
                this->message_converter.get_initial_timestamp();
            int time_step_size = this->message_converter.get_time_step_size();

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

//                    size_t timestamp =
//                        mission_init +
//                        (time_step + base_estimator->get_inference_horizon()) *
//                            time_step_size;
//                    message["Timestamp"] = timestamp;

                    message["Timestamp"] = "TODO";


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
    } // namespace model
} // namespace tomcat
