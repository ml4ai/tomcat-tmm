#include "ASISTStudy2Agent.h"

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
        ASISTStudy2Agent::ASISTStudy2Agent(const string& id,
                                           const string& estimates_topic,
                                           const string& log_topic)
            : ASISTAgent(id, estimates_topic, log_topic) {
            this->create_estimators();
        }

        ASISTStudy2Agent::~ASISTStudy2Agent() {}

        //----------------------------------------------------------------------
        // Copy & Move constructors/assignments
        //----------------------------------------------------------------------
        ASISTStudy2Agent::ASISTStudy2Agent(const ASISTStudy2Agent& agent)
            : ASISTAgent(agent.id, agent.estimates_topic, agent.log_topic) {}

        ASISTStudy2Agent&
        ASISTStudy2Agent::operator=(const ASISTStudy2Agent& agent) {
            ASISTAgent::copy(agent);
            return *this;
        }

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        nlohmann::json ASISTStudy2Agent::get_header_section() const {
            nlohmann::json header;
            header["timestamp"] = header;
            header["message_type"] = "agent";
            header["version"] = "1.0";

            return header;
        }

        nlohmann::json ASISTStudy2Agent::get_msg_section(int data_point) const {
            nlohmann::json msg;
            msg["trial_id"] = this->evidence_metadata[data_point]["trial"];
            msg["experiment_id"] =
                this->evidence_metadata[data_point]["experiment_id"];
            msg["timestamp"] = this->get_current_timestamp();
            msg["source"] = "tomcat-tmm";
            msg["sub_type"] = "prediction:state";
            msg["version"] = "1.0";

            return msg;
        }

        nlohmann::json
        ASISTStudy2Agent::get_data_section(int time_step,
                                           int data_point) const {
            nlohmann::json data;
            const string& initial_timestamp =
                this->evidence_metadata[data_point]["initial_timestamp"];
            int elapsed_time =
                time_step *
                (int)this->evidence_metadata[data_point]["step_size"];

            data["created"] =
                this->get_elapsed_timestamp(initial_timestamp, elapsed_time);
            data["unique_id"] = "Generate unique id";
            // data_message["start"] = null; Estimates apply immediately
            data["duration"] =
                (int)this->evidence_metadata[data_point]["step_size"];
            data["subject"] = "Player's name/codiname";
            data["predicted_property"] = "xxx";
            data["prediction"] = "xxx";
            data["probability_type"] = "xxx";
            data["probability"] = "xxx";
            data["confidence_type"] = "xxx";
            data["confidence"] = "xxx";
            data["explanation"] = "xxx";

            return data;
        }

    } // namespace model
} // namespace tomcat
