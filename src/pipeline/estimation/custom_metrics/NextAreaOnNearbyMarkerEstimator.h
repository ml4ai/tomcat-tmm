#pragma once

#include "pipeline/estimation/SamplerEstimator.h"

namespace tomcat {
    namespace model {

        /**
         * Represents a metric to estimate the probability of being in a
         * specific area (room or hallway) after the player leaves the range of
         * detection of a marker block given samples generated a couple of time
         * steps ahead.
         */
        class NextAreaOnNearbyMarkerEstimator : public SamplerEstimator {
          public:
            inline const static std::string NAME = "NextAreaOnNearbyMarker";

            //------------------------------------------------------------------
            // Constructors & Destructor
            //------------------------------------------------------------------

            /**
             * Creates an instance of the next area metric.
             *
             * @param model: DBN
             * @param player_num: player number
             *
             */
            NextAreaOnNearbyMarkerEstimator(
                const std::shared_ptr<DynamicBayesNet>& model,
                const nlohmann::json& json_config);

            ~NextAreaOnNearbyMarkerEstimator();

            //------------------------------------------------------------------
            // Copy & Move constructors/assignments
            //------------------------------------------------------------------

            NextAreaOnNearbyMarkerEstimator(
                const NextAreaOnNearbyMarkerEstimator& final_score);

            NextAreaOnNearbyMarkerEstimator&
            operator=(const NextAreaOnNearbyMarkerEstimator& final_Score);

            NextAreaOnNearbyMarkerEstimator(NextAreaOnNearbyMarkerEstimator&&) =
                default;

            NextAreaOnNearbyMarkerEstimator&
            operator=(NextAreaOnNearbyMarkerEstimator&&) = default;

            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            void prepare() override;

            std::string get_name() const override;

            /**
             * Estimate the final score based on the particles generated until
             * the end of the mission related to victim rescue.
             *
             * @param new_data: observations
             * @param particles: particles for the last time step processed
             * @param projected_particles: particles generated by transitioning
             * some time steps into the future
             * @param marginals: marginal distributions for nodes that were
             * rao-blackwellized
             * @param data_point_idx: index of the data point for which
             * particles were generated
             * @param time_step: time step when the first particles in de set
             * were generated
             */
            void estimate(const EvidenceSet& new_data,
                          const EvidenceSet& particles,
                          const EvidenceSet& projected_particles,
                          const EvidenceSet& marginals,
                          int data_point_idx,
                          int time_step) override;

            /**
             * Triggers an event every time the player enters the range of
             * detection of a marker block.
             *
             * @param data_point: data point
             * @param time_step: time step
             * @param new_data: test data
             *
             * @return
             */
            bool
            is_event_triggered_at(int data_point,
                                  int time_step,
                                  const EvidenceSet& new_data) const override;

            bool is_binary_on_prediction() const override;

            void prepare_for_the_next_data_point() const override;

            //------------------------------------------------------------------
            // Getters & Setters
            //------------------------------------------------------------------

            int get_player_number() const;

            int get_placed_by_player_nummber() const;

          private:
            //------------------------------------------------------------------
            // Data members
            //------------------------------------------------------------------

            int player_number;
            int placed_by_player_nummber;

            mutable bool within_marker_range;
            mutable int time_step_at_entrance;
            mutable int marker_at_entrance;
        };

    } // namespace model
} // namespace tomcat