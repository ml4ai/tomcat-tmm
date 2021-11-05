#include "EstimateReporter.h"

#include <boost/date_time/posix_time/posix_time.hpp>

#include "utils/EigenExtensions.h"

namespace tomcat {
    namespace model {

        using namespace std;
        namespace pt = boost::posix_time;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        EstimateReporter::EstimateReporter() {}

        EstimateReporter::~EstimateReporter() {}

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------

        nlohmann::json
        EstimateReporter::build_heartbeat_message(const AgentPtr& agent) {
            // No heartbeat message by default
            return {};
        }

        nlohmann::json EstimateReporter::build_start_of_mission_message(
            const AgentPtr& agent) {
            // No message by default
            return {};
        }

        nlohmann::json
        EstimateReporter::build_end_of_mission_message(const AgentPtr& agent) {
            // No message by default
            return {};
        }

        nlohmann::json EstimateReporter::build_message_by_request(
            const AgentPtr& agent,
            const nlohmann::json& request_message,
            int time_step) {
            // No message by default
            return {};
        }

        string EstimateReporter::get_request_response_topic(
            const nlohmann::json& request_message,
            const MessageBrokerConfiguration& broker_config) {
            // No topic by default
            return {};
        }

        void EstimateReporter::prepare() {

        }

        string EstimateReporter::get_current_timestamp() const {
            pt::ptime time = pt::microsec_clock::universal_time();
            return pt::to_iso_extended_string(time) + "Z";
        }

        string
        EstimateReporter::get_elapsed_timestamp(const string& initial_timestamp,
                                                int elapsed_time) const {
            tm t{};
            istringstream ss(initial_timestamp);

            // The precision of the timestamp will be in seconds.
            // milliseconds are ignored. This can be reaccessed
            // later if necessary. The milliseconds could be stored
            // in a separate attribute of this class.
            ss >> get_time(&t, "%Y-%m-%dT%T");
            string elapsed_timestamp;
            if (!ss.fail()) {
                time_t base_timestamp = mktime(&t);
                time_t timestamp = base_timestamp + elapsed_time;
                stringstream ss;
                ss << put_time(localtime(&timestamp), "%Y-%m-%dT%T.000Z");
                elapsed_timestamp = ss.str();
            }

            return elapsed_timestamp;
        }

    } // namespace model
} // namespace tomcat
