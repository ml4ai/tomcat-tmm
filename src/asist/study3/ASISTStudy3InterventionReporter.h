#pragma once

#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "asist/ASISTReporter.h"
#include "pipeline/estimation/Agent.h"

namespace tomcat {
    namespace model {

        /**
         * Represents an intervention reporter for study 3.
         */
        class ASISTStudy3InterventionReporter : public ASISTReporter {
          public:
            inline static const std::string NAME = "asist_study3_reporter";

            //------------------------------------------------------------------
            // Constructors & Destructor
            //------------------------------------------------------------------

            ASISTStudy3InterventionReporter(
                const nlohmann::json& json_settings);

            ~ASISTStudy3InterventionReporter() = default;

            //------------------------------------------------------------------
            // Copy & Move constructors/assignments
            //------------------------------------------------------------------
            ASISTStudy3InterventionReporter(
                const ASISTStudy3InterventionReporter& reporter);

            ASISTStudy3InterventionReporter&
            operator=(const ASISTStudy3InterventionReporter& reporter);

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
            // Static functions
            //------------------------------------------------------------------

            /**
             * Return a string with the participant ids separated by comma.
             *
             * @param agent: Agent
             * @param data_point: index of the trial being processed, if
             * estimates are being computed for multiple trials.
             *
             * @return: List of participant ids
             */
            static nlohmann::json get_player_list(const AgentPtr& agent,
                                                  int data_point);

            /**
             * Adds values to fields common to all interventions in the data
             * section of an intervention message.
             *
             * @param message: json message
             * @param agent: ASI
             * @param time_step: time step at which the message is being
             * generated
             * @param data_point: index of the trial being processed, if
             * estimates are being computed for multiple trials.
             */
            static void add_common_data_section(nlohmann::json& message,
                                                const AgentPtr& agent,
                                                int time_step,
                                                int data_point);

            /**
             * Creates a template intervention message.
             *
             * @param agent: ASI
             * @param time_step: time step at which the message is being
             * generated
             * @param data_point: index of the trial being processed, if
             * estimates are being computed for multiple trials.
             *
             * @return: json intervention message
             */
            static nlohmann::json get_template_intervention_message(
                const AgentPtr& agent, int time_step, int data_point);

            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            /**
             * Copy contents from another reporter.
             *
             * @param reporter: reporter
             */
            void copy(const ASISTStudy3InterventionReporter& reporter);

            /**
             * Handles intervention of type "Introduction"
             *
             * @param agent: ASI
             * @param time_step: time step of the intervention
             * @param messages: list of intervention messages in the time step
             */
            void
            intervene_on_introduction(const AgentPtr& agent,
                                      int time_step,
                                      std::vector<nlohmann::json>& messages);

            /**
             * Handles intervention of type "Motivation"
             *
             * @param agent: ASI
             * @param time_step: time step of the intervention
             * @param messages: list of intervention messages in the time step
             */
            void intervene_on_motivation(const AgentPtr& agent,
                                         int time_step,
                                         std::vector<nlohmann::json>& messages);

            /**
             * Handles intervention of type "Communication Marker"
             *
             * @param agent: ASI
             * @param time_step: time step of the intervention
             * @param messages: list of intervention messages in the time step
             */
            void intervene_on_communication_marker(
                const AgentPtr& agent,
                int time_step,
                std::vector<nlohmann::json>& messages);

            /**
             * Assembles ToMCAT's introduction as an intervention.
             *
             * @param agent: ASI
             * @param time_step: time step at which the message is being
             * generated
             * @param data_point: index of the trial being processed, if
             * estimates are being computed for multiple trials.
             *
             * @return: json intervention message
             */
            nlohmann::json get_introductory_intervention_message(
                const AgentPtr& agent, int time_step, int data_point) const;

            /**
             * Assembles motivation intervention.
             *
             * @param agent: ASI
             * @param time_step: time step at which the message is being
             * generated
             * @param data_point: index of the trial being processed, if
             * estimates are being computed for multiple trials.
             *
             * @return: json intervention message
             */
            nlohmann::json get_motivation_intervention_message(
                const AgentPtr& agent, int time_step, int data_point) const;

            /**
             * Assembles communication marker intervention.
             *
             * @param agent: ASI
             * @param time_step: time step at which the message is being
             * generated
             * @param data_point: index of the trial being processed, if
             * estimates are being computed for multiple trials.
             *
             * @return: json intervention message
             */
            nlohmann::json get_communication_marker_intervention_message(
                const AgentPtr& agent, int time_step, int data_point) const;

            //------------------------------------------------------------------
            // Data member
            //------------------------------------------------------------------
            bool introduced = false;
            bool intervened_on_motivation = false;
        };

    } // namespace model
} // namespace tomcat
