#pragma once

#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "asist/ASISTReporter.h"
#include "asist/study3/ASISTStudy3InterventionLogger.h"
#include "asist/study3/ASISTStudy3MessageConverter.h"
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

            void prepare() override;

            void set_logger(const OnlineLoggerPtr& logger) override;

          private:
            //------------------------------------------------------------------
            // Static functions
            //------------------------------------------------------------------

            /**
             * Return a string with the participant ids separated by comma.
             *
             * @param agent: Agent
             *
             * @return: List of participant ids
             */
            static nlohmann::json get_player_list(const AgentPtr& agent);

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
             * Returns the player color to add to a message based on its order.
             * The map follows the RGB order. That is 0 - Red, 1 - Green, 2 -
             * Blue
             *
             * @param player_order: player's index in a list
             *
             * @return Player's color (callsign)
             */
            static std::string player_order_to_color(int player_order);

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
             * Stores player info for quick retrieval. It assumes the agent does
             * not change over the course of a mission. Which is true for the
             * current implementation.
             *
             * @param agent: ASI
             */
            void store_player_info(const AgentPtr& agent);

            /**
             * Creates a template intervention message.
             *
             * @param agent: ASI
             * @param time_step: time step at which the message is being
             * generated
             * @param receivers: list of players that must receive the
             * intervention
             * @param intervention_type: type of the intervention matching the
             * types in the reporter settings file
             *
             * @return: json intervention message
             */
            nlohmann::json get_template_intervention_message(
                const AgentPtr& agent,
                int time_step,
                const nlohmann::json& receivers,
                const std::string& intervention_type) const;

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
             * Handles intervention of type "Ask for Help"
             *
             * @param agent: ASI
             * @param time_step: time step of the intervention
             * @param messages: list of intervention messages in the time step
             */
            void
            intervene_on_ask_for_help(const AgentPtr& agent,
                                      int time_step,
                                      std::vector<nlohmann::json>& messages);

            /**
             * Handles intervention of type "Help on the Way"
             *
             * @param agent: ASI
             * @param time_step: time step of the intervention
             * @param messages: list of intervention messages in the time step
             */
            void
            intervene_on_help_on_the_way(const AgentPtr& agent,
                                         int time_step,
                                         std::vector<nlohmann::json>& messages);

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
             * @param CDF: cdf that triggered intervention
             *
             * @return: json intervention message
             */
            nlohmann::json get_motivation_intervention_message(
                const AgentPtr& agent, int time_step, double cdf) const;

            /**
             * Assembles communication marker intervention.
             *
             * @param agent: ASI
             * @param time_step: time step at which the message is being
             * generated
             * @param player_order: player's index
             * @param marker: unspoken marker
             *
             * @return: json intervention message
             */
            nlohmann::json get_communication_marker_intervention_message(
                const AgentPtr& agent,
                int time_step,
                int player_order,
                const ASISTStudy3MessageConverter::Marker& marker) const;

            /**
             * Assembles ask-for-help (critical victim) intervention.
             *
             * @param agent: ASI
             * @param time_step: time step at which the message is being
             * generated
             * @param player_order: player's index
             *
             * @return: json intervention message
             */
            nlohmann::json
            get_ask_for_help_critical_victim_intervention_message(
                const AgentPtr& agent, int time_step, int player_order) const;

            /**
             * Assembles ask-for-help (threat) intervention.
             *
             * @param agent: ASI
             * @param time_step: time step at which the message is being
             * generated
             * @param player_order: player's index
             *
             * @return: json intervention message
             */
            nlohmann::json get_ask_for_help_threat_intervention_message(
                const AgentPtr& agent, int time_step, int player_order) const;

            /**
             * Assembles help-on-the-way intervention.
             *
             * @param agent: ASI
             * @param time_step: time step at which the message is being
             * generated
             * @param player_order: player's index
             *
             * @return: json intervention message
             */
            nlohmann::json get_help_on_the_way_intervention_message(
                const AgentPtr& agent, int time_step, int player_order) const;

            //------------------------------------------------------------------
            // Data member
            //------------------------------------------------------------------

            std::shared_ptr<ASISTStudy3InterventionLogger> custom_logger;

            // Player id per each RGB color.
            std::vector<std::string> player_ids_per_color;

            // Json objects with a list of player ids to address the whole team.
            nlohmann::json player_ids;

            bool player_info_initialized = false;
            bool introduced = false;
            bool intervened_on_motivation = false;
        };

    } // namespace model
} // namespace tomcat
