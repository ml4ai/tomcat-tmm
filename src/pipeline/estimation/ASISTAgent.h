#pragma once

#include <memory>
#include <string>
#include <vector>

#include "converter/ASISTMessageConverter.h"
#include "pipeline/estimation/Agent.h"

namespace tomcat {
    namespace model {

        /**
         * Represents an agent capable of communicating with the ASIST
         * program test bed.
         */
        class ASISTAgent : public Agent {
          public:
            //------------------------------------------------------------------
            // Constructors & Destructor
            //------------------------------------------------------------------

            /**
             * Creates an ASIST agent with a given ID.
             *
             * @param id: agent's ID
             * @param estimates_topic: message topic where estimates must be
             * published to
             * @param log_topic: message topic where processing log must be
             * published to
             * @param message_converter: converter used to extract data from
             * test bed messages
             */
            ASISTAgent(const std::string& id,
                       const std::string& estimates_topic,
                       const std::string& log_topic,
                       const ASISTMessageConverter& message_converter);

            virtual ~ASISTAgent();

            //------------------------------------------------------------------
            // Copy & Move constructors/assignments
            //------------------------------------------------------------------
            ASISTAgent(const ASISTAgent& agent);

            ASISTAgent& operator=(const ASISTAgent& agent);

            ASISTAgent(ASISTAgent&&) = default;

            ASISTAgent& operator=(ASISTAgent&&) = default;

            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------
            EvidenceSet message_to_data(const nlohmann::json& message) override;

            nlohmann::json estimates_to_message(
                const std::vector<std::shared_ptr<Estimator>>& estimators,
                int time_step) const override;

            std::unordered_set<std::string>
            get_topics_to_subscribe() const override;

            nlohmann::json build_log_message(const std::string& log) const
            override;

          private:
            //------------------------------------------------------------------
            // Data members
            //------------------------------------------------------------------
            ASISTMessageConverter message_converter;
        };

    } // namespace model
} // namespace tomcat
