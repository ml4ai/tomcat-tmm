#include "ASISTReporter.h"

#include <fmt/format.h>

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Copy & Move constructors/assignments
        //----------------------------------------------------------------------
        ASISTReporter::ASISTReporter(const ASISTReporter& reporter) {}

        ASISTReporter& ASISTReporter::operator=(const ASISTReporter& reporter) {
            return *this;
        }

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        void ASISTReporter::add_header_section(nlohmann::json& message,
                                               const AgentPtr& agent,
                                               const string& message_type,
                                               int time_step) const {

            message["timestamp"] = this->get_timestamp_at(agent, time_step);
            message["message_type"] = message_type;
            message["version"] = agent->get_version();
        }

        void ASISTReporter::add_msg_section(nlohmann::json& message,
                                            const AgentPtr& agent,
                                            const string& sub_type,
                                            int time_step) const {

            message["trial_id"] =
                agent->get_evidence_metadata()["trial_unique_id"];
            message["experiment_id"] =
                agent->get_evidence_metadata()["experiment_id"];
            message["timestamp"] = this->get_timestamp_at(agent, time_step);
            message["source"] = agent->get_id();
            message["version"] = "1.0";
            message["trial_number"] =
                agent->get_evidence_metadata()["trial_id"];
            message["sub_type"] = sub_type;
        }

        string ASISTReporter::get_timestamp_at(const AgentPtr& agent,
                                               int time_step) const {
            const string& initial_timestamp =
                agent->get_evidence_metadata()["initial_timestamp"];
            int elapsed_time =
                time_step * (int)agent->get_evidence_metadata()["step_size"];

            return this->get_elapsed_timestamp(initial_timestamp, elapsed_time);
        }

        int ASISTReporter::get_milliseconds_at(const AgentPtr& agent,
                                               int time_step) const {
            int step_size = agent->get_evidence_metadata()["step_size"];
            return time_step * step_size * 1000;
        }
    } // namespace model
} // namespace tomcat
