#include "EstimateReporter.h"

#include <boost/date_time/posix_time/posix_time.hpp>

#include "utils/EigenExtensions.h"
#include "asist/study2/ASISTStudy2EstimateReporter.h"
#include "asist/study3/ASISTStudy3InterventionReporter.h"

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
        // Static functions
        //----------------------------------------------------------------------

        EstimateReporterPtr
        factory(const std::string& reporter_name) {
            EstimateReporterPtr reporter;

            if (reporter_name == ASISTStudy2EstimateReporter::NAME) {
                reporter = make_shared<ASISTStudy2EstimateReporter>();
            } else if (reporter_name == ASISTStudy3InterventionReporter::NAME) {
                reporter = make_shared<ASISTStudy3InterventionReporter>();
            }

            return reporter;
        }

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------

        void EstimateReporter::copy(const EstimateReporter& reporter) {
            this->json_settings = reporter.json_settings;
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

        //----------------------------------------------------------------------
        // Getters & Setters
        //----------------------------------------------------------------------

        void EstimateReporter::set_json_settings(
            const nlohmann::json& json_settings) {
            EstimateReporter::json_settings = json_settings;
        }

    } // namespace model
} // namespace tomcat
