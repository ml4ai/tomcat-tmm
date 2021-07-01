#pragma once

#include <memory>
#include <string>
#include <vector>

#include "pipeline/estimation/ASISTEstimateReporter.h"

namespace tomcat {
    namespace model {

        /**
         * Represents a TMM reporter for study 2 of the ASIST program.
         */
        class ASISTStudy2EstimateReporter : public ASISTEstimateReporter {
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

          protected:
            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            nlohmann::json
            get_header_section(const AgentPtr& agent) const override;

            std::pair<nlohmann::json, nlohmann::json>
            get_msg_section(const AgentPtr& agent,
                            int data_point) const override;

            std::pair<nlohmann::json, nlohmann::json>
            get_data_section(const AgentPtr& agent,
                             int time_step,
                             int data_point) const override;

          private:
            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            /**
             * Adds information about the prediction of the final team score to
             * the list of predictions.
             *
             * @param json_predictions: message with list of predictions
             * @param agent: agent responsible for the predictions
             * @param estimator: final team score estimator
             * @param time_step: time step of the prediction
             * @param data_point: mission trial index (if multiple missions are
             * being processed at the same time)
             */
            void
            add_final_team_score_prediction(nlohmann::json& json_predictions,
                                            const AgentPtr& agent,
                                            const EstimatorPtr& estimator,
                                            int time_step,
                                            int data_point) const;

            /**
             * Adds information about the prediction of the knowledge about the
             * map info per player to the list of predictions.
             *
             * @param json_predictions: message with list of predictions
             * @param agent: agent responsible for the predictions
             * @param estimator: map info estimator
             * @param time_step: time step of the prediction
             * @param data_point: mission trial index (if multiple missions are
             * being processed at the same time)
             */
            void add_map_info_prediction(nlohmann::json& json_predictions,
                                         const AgentPtr& agent,
                                         const EstimatorPtr& estimator,
                                         int time_step,
                                         int data_point) const;

            /**
             * Adds information about the prediction of the knowledge about the
             * marker legend per player to the list of predictions.
             *
             * @param json_predictions: message with list of predictions
             * @param agent: agent responsible for the predictions
             * @param estimator: marker legend estimator
             * @param time_step: time step of the prediction
             * @param data_point: mission trial index (if multiple missions are
             * being processed at the same time)
             */
            void add_marker_legend_prediction(nlohmann::json& json_predictions,
                                              const AgentPtr& agent,
                                              const EstimatorPtr& estimator,
                                              int time_step,
                                              int data_point) const;

            /**
             * Adds information about the prediction of action taken in face of
             * a false belief per player to the list of predictions.
             *
             * @param json_predictions: message with list of predictions
             * @param agent: agent responsible for the predictions
             * @param estimator: marker false belief estimator
             * @param time_step: time step of the prediction
             * @param data_point: mission trial index (if multiple missions are
             * being processed at the same time)
             */
            void
            add_marker_false_belief_prediction(nlohmann::json& json_predictions,
                                               const AgentPtr& agent,
                                               const EstimatorPtr& estimator,
                                               int time_step,
                                               int data_point) const;
        };

    } // namespace model
} // namespace tomcat
