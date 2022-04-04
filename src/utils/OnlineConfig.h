#pragma once

#include <string>

namespace tomcat {
    namespace model {
        /**
         * This struct contains information needed to connect to a message
         * broker to either subscribe or publish to a topic.
         */
        struct MessageBrokerConfiguration {
            int timeout = 9999;
            std::string address;
            int port;
            int num_connection_trials;
            int milliseconds_before_retrial;
            std::string intervention_topic;
        };
    } // namespace model
} // namespace tomcat