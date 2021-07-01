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
        ASISTEstimateReporter::estimates_to_message(const AgentPtr& agent,
                                                    int time_step) const {
            vector<nlohmann::json> messages;

            nlohmann::json message;
            message["header"] = move(this->get_header_section(agent));
            int num_data_points = agent->get_evidence_metadata().size();
            for (int d = 0; d < num_data_points; d++) {
                message["msg"] = move(this->get_msg_section(agent, d));
                message["data"] =
                    move(this->get_data_section(agent, time_step, d));
                messages.push_back(message);
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
