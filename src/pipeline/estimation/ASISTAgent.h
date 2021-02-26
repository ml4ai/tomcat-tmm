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
             */
            ASISTAgent(const std::string& id,
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
            EvidenceSet
            message_to_data(const nlohmann::json& message) override;

            nlohmann::json estimates_to_message(
                const std::vector<std::shared_ptr<Estimator>>& estimators,
                int time_step) const override;

          private:
            //------------------------------------------------------------------
            // Data members
            //------------------------------------------------------------------
            ASISTMessageConverter message_converter;
        };

    } // namespace model
} // namespace tomcat
