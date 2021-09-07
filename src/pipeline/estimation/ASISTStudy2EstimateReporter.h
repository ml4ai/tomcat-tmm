#pragma once

#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "pipeline/estimation/Agent.h"
#include "pipeline/estimation/EstimateReporter.h"
#include "pipeline/estimation/custom_metrics/FinalTeamScoreEstimator.h"
#include "pipeline/estimation/custom_metrics/IndependentMapVersionAssignmentEstimator.h"
#include "pipeline/estimation/custom_metrics/NextAreaOnNearbyMarkerEstimator.h"

namespace tomcat {
    namespace model {

        /**
         * Represents a TMM reporter for study 2 of the ASIST program.
         */
        class ASISTStudy2EstimateReporter : public EstimateReporter {
          public:
            //------------------------------------------------------------------
            // Constructors & Destructor
            //------------------------------------------------------------------

            ASISTStudy2EstimateReporter();

            virtual ~ASISTStudy2EstimateReporter();

            //------------------------------------------------------------------
            // Copy & Move constructors/assignments
            //------------------------------------------------------------------
            ASISTStudy2EstimateReporter(
                const ASISTStudy2EstimateReporter& agent);

            ASISTStudy2EstimateReporter&
            operator=(const ASISTStudy2EstimateReporter& agent);

            ASISTStudy2EstimateReporter(ASISTStudy2EstimateReporter&&) =
                default;

            ASISTStudy2EstimateReporter&
            operator=(ASISTStudy2EstimateReporter&&) = default;

            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------
            std::vector<nlohmann::json>
            translate_estimates_to_messages(const AgentPtr& agent,
                                            int time_step) const;

            nlohmann::json build_log_message(const AgentPtr& agent,
                                             const std::string& log) const;

          protected:
            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            nlohmann::json get_header_section(const AgentPtr& agent) const;

            nlohmann::json get_common_msg_section(const AgentPtr& agent,
                                                  int data_point) const;

            nlohmann::json get_common_data_section(const AgentPtr& agent,
                                                   int data_point) const;

          private:
            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            /**
             * Adds information about the prediction of the final team score to
             * the list of predictions.
             *
             * @param agent: agent responsible for the predictions
             * @param estimator: final team score estimator
             * @param time_step: time step of the prediction
             * @param data_point: mission trial index (if multiple missions are
             * being processed at the same time)
             */
            std::vector<nlohmann::json> get_final_team_score_predictions(
                const AgentPtr& agent,
                const std::shared_ptr<FinalTeamScoreEstimator>& estimator,
                int time_step,
                int data_point) const;

            /**
             * Adds information about the prediction of the knowledge about the
             * map info per player to the list of predictions.
             *
             * @param agent: agent responsible for the predictions
             * @param estimator: map info estimator
             * @param time_step: time step of the prediction
             * @param data_point: mission trial index (if multiple missions are
             * being processed at the same time)
             */
            std::vector<nlohmann::json> get_map_info_predictions(
                const AgentPtr& agent,
                const std::shared_ptr<IndependentMapVersionAssignmentEstimator>&
                    estimator,
                int time_step,
                int data_point) const;

            /**
             * Adds information about the prediction of the knowledge about the
             * marker legend per player to the list of predictions.
             *
             * @param agent: agent responsible for the predictions
             * @param estimator: marker legend estimator
             * @param time_step: time step of the prediction
             * @param data_point: mission trial index (if multiple missions are
             * being processed at the same time)
             */
            std::vector<nlohmann::json> get_marker_legend_predictions(
                const AgentPtr& agent,
                const EstimatorPtr& estimator,
                int time_step,
                int data_point) const;

            /**
             * Adds information about the prediction of action taken in face of
             * a false belief per player to the list of predictions.
             *
             * @param agent: agent responsible for the predictions
             * @param estimator: marker false belief estimator
             * @param time_step: time step of the prediction
             * @param data_point: mission trial index (if multiple missions are
             * being processed at the same time)
             */
            std::vector<nlohmann::json> get_next_area_predictions(
                const AgentPtr& agent,
                const std::shared_ptr<NextAreaOnNearbyMarkerEstimator>&
                    estimator,
                int time_step,
                int data_point) const;

            /**
             * Calculates the timestamp at a given time step within the mission.
             *
             * @param agent: agent responsible for the predictions
             * @param time_step: time step
             * @param data_point: mission trial index (if multiple missions are
             * being processed at the same time)
             *
             * @return Initial timestamp + elapsed time
             */
            std::string get_timestamp_at(const AgentPtr& agent,
                                         int time_step,
                                         int data_point) const;

            /**
             * Calculates the milliseconds at a given time step within the
             * mission.
             *
             * @param agent: agent responsible for the predictions
             * @param time_step: time step
             * @param data_point: mission trial index (if multiple missions are
             * being processed at the same time)
             *
             * @return Time step * step size * 1000
             */
            int get_milliseconds_at(const AgentPtr& agent,
                                    int time_step,
                                    int data_point) const;

            /**
             * Store event info from the evidence set metadata the first time an
             * M7 prediction is reported.
             *
             * @param evidence_metadata: metadata of the evidence set containing
             * details of all M7 events in that set.
             */
            void initialize_m7(const nlohmann::json& evidence_metadata) const;

            //------------------------------------------------------------------
            // Data member
            //------------------------------------------------------------------

            mutable bool m7_initialized = false;
            mutable std::vector<
                std::unordered_map<int, std::vector<nlohmann::json>>>
                m7_time_steps_per_data_point;
        };

    } // namespace model
} // namespace tomcat
