#include "ASISTReporter.h"

#include <fmt/format.h>

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------

        ASISTReporter::ASISTReporter(const nlohmann::json& json_settings)
            : EstimateReporter(json_settings) {}

        //----------------------------------------------------------------------
        // Copy & Move constructors/assignments
        //----------------------------------------------------------------------
        ASISTReporter::ASISTReporter(const ASISTReporter& reporter)
            : EstimateReporter(reporter.json_settings) {}

        ASISTReporter& ASISTReporter::operator=(const ASISTReporter& reporter) {
            return *this;
        }

        //----------------------------------------------------------------------
        // Static functions
        //----------------------------------------------------------------------
        void ASISTReporter::add_header_section(nlohmann::json& message,
                                               const AgentPtr& agent,
                                               const string& message_type,
                                               int time_step,
                                               int data_point) {

            message["timestamp"] = get_timestamp_at(agent, time_step, data_point);
            message["message_type"] = message_type;
            message["version"] = agent->get_version();
        }

        void ASISTReporter::add_msg_section(nlohmann::json& message,
                                            const AgentPtr& agent,
                                            const string& sub_type,
                                            int time_step,
                                            int data_point) {

            message["trial_id"] =
                agent->get_evidence_metadata()[data_point]["trial_id"];
            message["experiment_id"] =
                agent->get_evidence_metadata()[data_point]["experiment_id"];
            message["timestamp"] = get_timestamp_at(agent, time_step, data_point);
            message["source"] = agent->get_id();
            message["version"] = "1.0";
            message["trial_number"] =
                agent->get_evidence_metadata()[data_point]["trial"];
            message["sub_type"] = sub_type;
        }

        string ASISTReporter::get_timestamp_at(const AgentPtr& agent,
                                               int time_step,
                                               int data_point) {
            const string& initial_timestamp =
                agent->get_evidence_metadata()[data_point]["initial_timestamp"];
            int elapsed_time =
                time_step * (int)agent->get_evidence_metadata()[data_point]["step_size"];

            return get_elapsed_timestamp(initial_timestamp, elapsed_time);
        }

        int ASISTReporter::get_milliseconds_at(const AgentPtr& agent,
                                               int time_step,
                                               int data_point) {

            int step_size =
                agent->get_evidence_metadata()[data_point]["step_size"];
            return time_step * step_size * 1000;
        }
    } // namespace model
} // namespace tomcat
