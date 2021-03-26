#include "Agent.h"

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        Agent::Agent(const string& id,
                     const string& estimates_topic,
                     const string& log_topic)
            : id(id), estimates_topic(estimates_topic), log_topic(log_topic) {}

        Agent::~Agent() {}

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------

        void Agent::copy(const Agent& agent) {
            this->id = agent.id;
            this->estimates_topic = agent.estimates_topic;
            this->log_topic = agent.log_topic;
        }

        //----------------------------------------------------------------------
        // Getters & Setters
        //----------------------------------------------------------------------
        const string& Agent::get_id() const { return id; }

        const string& Agent::get_estimates_topic() const {
            return estimates_topic;
        }
        const string& Agent::get_log_topic() const { return log_topic; }
    } // namespace model
} // namespace tomcat
