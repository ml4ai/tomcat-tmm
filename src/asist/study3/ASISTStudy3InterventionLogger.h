#pragma once

#include <memory>

#include "asist/study3/ASISTStudy3MessageConverter.h"
#include "pgm/EvidenceSet.h"
#include "pipeline/estimation/Agent.h"
#include "pipeline/estimation/OnlineLogger.h"
#include "reporter/EstimateReporter.h"
#include "utils/Definitions.h"

namespace tomcat {
    namespace model {

        class ASISTStudy3InterventionLogger : public OnlineLogger {
          public:
            inline static const std::string NAME =
                "asist_study3_intervention_logger";

            //------------------------------------------------------------------
            // Constructors & Destructor
            //------------------------------------------------------------------

            ASISTStudy3InterventionLogger(const std::string& log_filepath);

            ~ASISTStudy3InterventionLogger() = default;

            //------------------------------------------------------------------
            // Copy & Move constructors/assignments
            //------------------------------------------------------------------

            ASISTStudy3InterventionLogger(
                const ASISTStudy3InterventionLogger&) = delete;

            ASISTStudy3InterventionLogger&
            operator=(const ASISTStudy3InterventionLogger&) = delete;

            ASISTStudy3InterventionLogger(ASISTStudy3InterventionLogger&&) =
                default;

            ASISTStudy3InterventionLogger&
            operator=(ASISTStudy3InterventionLogger&&) = default;

            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            void create_header() override;

            void log(const std::string& text) override;

            void log_first_evidence_set(const EvidenceSet& data) override;

            void log_watch_intervention(int time_step, const std::string& text);

            void log_intervene_on_introduction(int time_step);

            void log_intervene_on_motivation(int time_step,
                                             int num_encouragements,
                                             double probability);

            void log_watch_motivation_intervention(int time_step);

            void log_cancel_motivation_intervention(int time_step,
                                                    int num_encouragements,
                                                    double probability);

            void log_update_motivation_intervention(int time_step,
                                                    int num_encouragements);

            void log_empty_encouragements(int time_step);

            void log_intervene_on_marker(int time_step, int player_order);

            void log_cancel_communication_marker_intervention(
                int time_step,
                int player_order,
                const ASISTStudy3MessageConverter::Marker& marker,
                bool speech,
                bool marker_removal);

            void log_watch_communication_marker_intervention(
                int time_step,
                int player_order,
                const ASISTStudy3MessageConverter::Marker& marker);

            void log_activate_communication_marker_intervention(
                int time_step,
                int player_order,
                const ASISTStudy3MessageConverter::Marker& active_marker,
                bool area_changed,
                bool victim_interaction,
                bool marker_placed);

            void
            log_intervene_on_ask_for_help_critical_victim(int time_step,
                                                          int player_order);

            void log_watch_ask_for_help_critical_victim_intervention(
                int time_step, int player_order);

            void log_activate_ask_for_help_critical_victim_intervention(
                int time_step, int player_order, int latency);

            void log_cancel_ask_for_help_critical_victim_intervention(
                int time_step,
                int player_order,
                bool area_changed,
                bool help_request,
                bool mention_to_critical_victim,
                bool other_players_around);

            void
            log_intervene_on_ask_for_help_threat(int time_step,
                                                          int player_order);

            void log_watch_ask_for_help_threat_intervention(
                int time_step, int player_order);

            void log_activate_ask_for_help_threat_intervention(
                int time_step, int player_order, int latency);

            void log_cancel_ask_for_help_threat_intervention(
                int time_step,
                int player_order,
                bool area_changed,
                bool help_request,
                bool being_released);

            void log_cancel_intervention(int time_step,
                                         const std::string& text);

            void log_activate_intervention(int time_step,
                                           const std::string& text);

            void log_trigger_intervention(int time_step,
                                          const std::string& text);

            void log(int time_step, const std::string& text);

            //------------------------------------------------------------------
            // Getters & Setters
            //------------------------------------------------------------------

            void set_time_step_size(int time_step_size);
            void set_num_time_steps(int num_time_steps);

          private:
            inline const static std::vector<std::string> PLAYER_ORDER_TO_COLOR =
                {"Red", "Green", "Blue"};

            //------------------------------------------------------------------
            // Static functions
            //------------------------------------------------------------------

            static std::string change_extension(const std::string& filepath);

            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            void log_common_info(const std::string& mission_timer,
                                 const std::string& text);

            std::string time_step_to_mission_timer(int time_step) const;

            //------------------------------------------------------------------
            // Data members
            //------------------------------------------------------------------

            int time_step_size = 1;
            int num_time_steps = 0;
        };

    } // namespace model
} // namespace tomcat
