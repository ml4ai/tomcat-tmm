#pragma once

#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "asist/study3/ASISTReporter.h"
#include "pipeline/estimation/Agent.h"

namespace tomcat {
    namespace model {

        /**
         * Represents an intervention reporter for study 3.
         */
        class ASISTStudy3InterventionReporter : public ASISTReporter {
          public:
            //------------------------------------------------------------------
            // Constructors & Destructor
            //------------------------------------------------------------------

            ASISTStudy3InterventionReporter(const nlohmann::json& settings);

            ~ASISTStudy3InterventionReporter() = default;

            //------------------------------------------------------------------
            // Copy & Move constructors/assignments
            //------------------------------------------------------------------
            ASISTStudy3InterventionReporter(
                const ASISTStudy3InterventionReporter& agent);

            ASISTStudy3InterventionReporter&
            operator=(const ASISTStudy3InterventionReporter& agent);

            ASISTStudy3InterventionReporter(ASISTStudy3InterventionReporter&&) =
                default;

            ASISTStudy3InterventionReporter&
            operator=(ASISTStudy3InterventionReporter&&) = default;

            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------
            std::vector<nlohmann::json>
            translate_estimates_to_messages(const AgentPtr& agent,
                                            int time_step) override;

            nlohmann::json
            build_log_message(const AgentPtr& agent,
                              const std::string& log) const override;

            void prepare() override;

          private:
            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            void copy(const ASISTStudy3InterventionReporter& reporter);

            /**
             * Adds values to fields common to all interventions in the data
             * section of an intervention message.
             *
             * @param message: json message
             * @param agent: ASI
             * @param time_step: time step at which the message is being
             * generated
             */
            void add_common_data_section(nlohmann::json& message,
                                         const AgentPtr& agent,
                                         int time_step) const;

            /**
             * Creates a template intervention message.
             *
             * @param agent: ASI
             * @param time_step: time step at which the message is being
             * generated
             *
             * @return: json intervention message
             */
            nlohmann::json
            get_template_intervention_message(const AgentPtr& agent,
                                                  int time_step) const;

            /**
             * Assembles ToMCAT's introduction as an intervention.
             *
             * @param agent: ASI
             * @param time_step: time step at which the message is being
             * generated
             *
             * @return: json intervention message
             */
            nlohmann::json
            get_introductory_intervention_message(const AgentPtr& agent,
                                                  int time_step) const;

            /**
             * Assembles motivation intervention.
             *
             * @param agent: ASI
             * @param time_step: time step at which the message is being
             * generated
             *
             * @return: json intervention message
             */
            nlohmann::json
            get_motivation_intervention_message(const AgentPtr& agent,
                                                int time_step) const;

            /**
             * Return a string with the participant ids separated by comma.
             *
             * @param agent: Agent
             *
             * @return: List of participant ids
             */
            nlohmann::json get_player_list(const AgentPtr& agent) const;

            //------------------------------------------------------------------
            // Data member
            //------------------------------------------------------------------
            nlohmann::json settings;
            bool introduced;
            bool intervened_on_motivation;
        };

    } // namespace model
} // namespace tomcat
