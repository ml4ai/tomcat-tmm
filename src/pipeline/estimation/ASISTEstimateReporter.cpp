#include "ASISTEstimateReporter.h"

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        ASISTEstimateReporter::ASISTEstimateReporter() {}

        ASISTEstimateReporter::~ASISTEstimateReporter() {}

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        vector<nlohmann::json>
        ASISTEstimateReporter::translate_estimates_to_messages(const AgentPtr& agent,
                                                    int time_step) const {
            vector<nlohmann::json> messages;

            nlohmann::json message;
            message["header"] = move(this->get_header_section(agent));
            int num_data_points = agent->get_evidence_metadata().size();
            for (int d = 0; d < num_data_points; d++) {
                const auto& [msg_state, msg_prediction] = this->get_msg_section(agent, d);
                const auto& [data_state, data_prediction] = this->get_data_section(agent, time_step, d);

                if (!msg_state.empty() && !data_state.empty()) {
                    message["msg"] = move(msg_state);
                    message["data"] = move(data_state);
                    messages.push_back(message);
                }

                if (!msg_prediction.empty() && !data_prediction.empty()) {
                    message["msg"] = move(msg_prediction);
                    message["data"] = move(data_prediction);
                    messages.push_back(message);
                }
            }

            return messages;
        }

        nlohmann::json
        ASISTEstimateReporter::build_log_message(const AgentPtr& agent,
                                                 const string& log) const {
            // No predefined format for the ASIST program
            nlohmann::json message;
            return message;
        }

    } // namespace model
} // namespace tomcat
