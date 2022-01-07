#pragma once

#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "pipeline/estimation/Agent.h"
#include "reporter/EstimateReporter.h"

namespace tomcat {
    namespace model {

        /**
         * Represents a TMM reporter for study 3 of the ASIST program.
         */
        class ASISTStudy3InterventionReporter : public EstimateReporter {
          public:
            //------------------------------------------------------------------
            // Constructors & Destructor
            //------------------------------------------------------------------

            ASISTStudy3InterventionReporter();

            virtual ~ASISTStudy3InterventionReporter();

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

          protected:
            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            nlohmann::json
            get_header_section(const AgentPtr& agent,
                               const std::string& message_type) const;

            nlohmann::json get_msg_section(const AgentPtr& agent,
                                           const std::string& sub_type) const;

            nlohmann::json get_common_data_section(const AgentPtr& agent,
                                                   int time_step);

          private:
            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            /**
             * Returns ToMCAT introductory speech.
             *
             * @return: Speech
             */
            std::string get_introductory_speech() const;

            /**
             * Returns ToMCAT closing speech.
             *
             * @return: Speech
             */
            std::string get_closing_speech() const;

            /**
             * Get estimated team quality and confidence of the estimate in a
             * given time step.
             *
             * @param estimator: team quality estimator
             * @param time_step: time step
             *
             * @return: pair of team quality and confidence
             */
            std::pair<int, double>
            get_estimated_team_quality(const EstimatorPtr& estimator,
                                       int time_step) const;

            /**
             * Get quality decay considering a specific player spends 30 more
             * seconds in the current section.
             *
             * @param estimator: team quality estimator
             * @param time_step: time step
             * @param player_number: player_number
             *
             * @return Quality decay
             */
            double get_team_quality_decay(const EstimatorPtr& estimator,
                                          int time_step,
                                          int player_number) const;

            /**
             * Gets ToMCAT's speech to the players about the quality of their
             * team.
             *
             * @param quality: bad, medium or good
             * @param confidence: ToMCAT's confidence about the team's quality.
             *
             * @return: Speech
             */
            std::string get_tomcat_team_quality_speech(int quality,
                                                       int confidence) const;

            /**
             * Gets ToMCAT's speech to the players alerting them about the
             * remaining time until the mission ending
             *
             * @param time_step: mission time step
             *
             * @return: Speech
             */
            std::string get_tomcat_timer_alert_speech(int time_step) const;

            /**
             * Gets ToMCAT's speech to a specific player that searching in this
             * area longer might decrease the team performance.
             *
             * @return: Speech
             */
            std::string get_tomcat_team_quality_decay_speech() const;

            /**
             * Return a string with the participant ids separated by comma.
             *
             * @param agent: Agent
             *
             * @return: List of participant ids
             */
            nlohmann::json get_player_list(const AgentPtr& agent) const;

            /**
             * Calculates the timestamp at a given time step within the mission.
             *
             * @param agent: agent responsible for the predictions
             * @param time_step: time step
             *
             * @return Initial timestamp + elapsed time
             */
            std::string get_timestamp_at(const AgentPtr& agent,
                                         int time_step) const;

            /**
             * Calculates the milliseconds at a given time step within the
             * mission.
             *
             * @param agent: agent responsible for the predictions
             * @param time_step: time step
             *
             * @return Time step * step size * 1000
             */
            int get_milliseconds_at(const AgentPtr& agent, int time_step) const;

            //------------------------------------------------------------------
            // Data member
            //------------------------------------------------------------------
            int last_quality = -1;

            int intro_time;
            int closing_time;
            std::unordered_set<int> mission_timer_alert_times;
            std::unordered_set<int> performance_feedback_times;
            std::unordered_set<int> map_section_check_times;
            std::unordered_set<int> all_intervention_times;
        };

    } // namespace model
} // namespace tomcat
