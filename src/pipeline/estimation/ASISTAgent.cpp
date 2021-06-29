#include "ASISTAgent.h"

#include <iomanip>
#include <time.h>

#include <boost/date_time/posix_time/posix_time.hpp>

#include "utils/EigenExtensions.h"

namespace pt = boost::posix_time;

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        ASISTAgent::ASISTAgent(const string& id,
                                           const string& estimates_topic,
                                           const string& log_topic)
            : Agent(id, estimates_topic, log_topic) {
        }

        ASISTAgent::~ASISTAgent() {}

        //----------------------------------------------------------------------
        // Copy & Move constructors/assignments
        //----------------------------------------------------------------------
        ASISTAgent::ASISTAgent(const ASISTAgent& agent)
            : Agent(agent.id, agent.estimates_topic, agent.log_topic) {}

        ASISTAgent&
        ASISTAgent::operator=(const ASISTAgent& agent) {
            Agent::copy(agent);
            return *this;
        }

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        vector<nlohmann::json>
        ASISTAgent::estimates_to_message(int time_step) const {
            vector<nlohmann::json> messages;

            nlohmann::json message;
            message["header"] = move(this->get_header_section());
            int num_data_points = this->evidence_metadata.size();
            for (int d = 0; d < num_data_points; d++) {
                nlohmann::json message;
                message["msg"] = move(this->get_msg_section(d));

                for (auto estimator : this->estimators) {
                    message["data"] =
                        move(this->get_data_section(time_step, d));
                    messages.push_back(message);
                }
            }
        }

        string ASISTAgent::get_current_timestamp() const {
            pt::ptime time = pt::microsec_clock::universal_time();
            return pt::to_iso_extended_string(time) + "Z";
        }

        string
        ASISTAgent::get_elapsed_timestamp(const string& initial_timestamp,
                                                int elapsed_time) const {
            tm t{};
            istringstream ss(initial_timestamp);

            // The precision of the timestamp will be in seconds.
            // milliseconds are ignored. This can be reaccessed
            // later if necessary. The milliseconds could be stored
            // in a separate attribute of this class.
            ss >> get_time(&t, "%Y-%m-%dT%T");
            if (!ss.fail()) {
                time_t initial_timestamp = mktime(&t);
                time_t elapsed_timestamp = initial_timestamp() + elapsed_time;
                stringstream ss;
                ss << put_time(localtime(&elapsed_timestamp),
                               "%Y-%m-%dT%T.000Z");
            }
            else {
                ss.clear();
            }

            return ss.str();
        }

        nlohmann::json
        ASISTAgent::build_log_message(const string& log) const {
            // No predefined format
            nlohmann::json message;
            return message;
        }

    } // namespace model
} // namespace tomcat
