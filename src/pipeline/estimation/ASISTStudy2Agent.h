#pragma once

#include <memory>
#include <string>
#include <vector>

#include "pipeline/estimation/ASISTAgent.h"

namespace tomcat {
    namespace model {

        /**
         * Represents a TMM agent for study 2 of the ASIST program.
         */
        class ASISTStudy2Agent : public ASISTAgent {
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
             */
            ASISTStudy2Agent(const std::string& id,
                             const std::string& estimates_topic,
                             const std::string& log_topic);

            virtual ~ASISTStudy2Agent();

            //------------------------------------------------------------------
            // Copy & Move constructors/assignments
            //------------------------------------------------------------------
            ASISTStudy2Agent(const ASISTStudy2Agent& agent);

            ASISTStudy2Agent& operator=(const ASISTStudy2Agent& agent);

            ASISTStudy2Agent(ASISTStudy2Agent&&) = default;

            ASISTStudy2Agent& operator=(ASISTStudy2Agent&&) = default;

          protected:
            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            nlohmann::json get_header_section() const override;

            nlohmann::json get_msg_section(int data_point) const override;

            nlohmann::json get_data_section(int time_step,
                                            int data_point) const override;

          private:
            /**
             * Creates estimators used by this particular agent. This agent
             * already knows the estimators it need for this specific study, so
             * they will be created here.
             */
            void create_estimators();
        };

    } // namespace model
} // namespace tomcat
