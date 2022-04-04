#pragma once

#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

#include "asist/study3/ASISTStudy3MessageConverter.h"
#include "pgm/EvidenceSet.h"
#include "pipeline/Model.h"
#include "pipeline/estimation/Estimator.h"
#include "pipeline/estimation/SumProductEstimator.h"
#include "utils/Definitions.h"

namespace tomcat::model {

    /**
     * This class represents an estimator to perform interventions in Study 3 of
     * the ASIST program. It consists of a collection of models to keep track of
     * different beliefs for different players in the mission.
     */
    class ASISTStudy3InterventionEstimator : public Estimator {
      public:
        inline static const std::string NAME =
            "asist_study3_intervention_estimator";

        //------------------------------------------------------------------
        // Constructors & Destructor
        //------------------------------------------------------------------

        explicit ASISTStudy3InterventionEstimator(const ModelPtr& model);

        ~ASISTStudy3InterventionEstimator() = default;

        //------------------------------------------------------------------
        // Copy & Move constructors/assignments
        //------------------------------------------------------------------
        ASISTStudy3InterventionEstimator(
            const ASISTStudy3InterventionEstimator& estimator);

        ASISTStudy3InterventionEstimator&
        operator=(const ASISTStudy3InterventionEstimator& estimator);

        ASISTStudy3InterventionEstimator(ASISTStudy3InterventionEstimator&&) =
            default;

        ASISTStudy3InterventionEstimator&
        operator=(ASISTStudy3InterventionEstimator&&) = default;

        //------------------------------------------------------------------
        // Member functions
        //------------------------------------------------------------------
        void estimate(const EvidenceSet& new_data) override;

        void get_info(nlohmann::json& json) const override;

        std::string get_name() const override;

        void prepare() override;

        /**
         * Get the CDF of the number of encouragement utterances identified in
         * mission 1.
         *
         * @return CDF
         */
        double get_encouragement_cdf();

        /**
         * Removes current active unspoken marker from the list.
         *
         * @param player_order: index of the player that has an active unspoken
         * marker
         */
        void clear_active_unspoken_marker(int player_order);

        //------------------------------------------------------------------
        // Getters & Setters
        //------------------------------------------------------------------

        //------------------------------------------------------------------
        // Virtual member functions
        //------------------------------------------------------------------

        //------------------------------------------------------------------
        // Getters & Setters
        //------------------------------------------------------------------

        int get_last_time_step() const;

        const std::vector<ASISTStudy3MessageConverter::Marker>&
        get_active_unspoken_markers() const;

      protected:
        //------------------------------------------------------------------
        // Member functions
        //------------------------------------------------------------------

        /**
         * Copies data member from another estimator.
         *
         * @param estimator: other estimator
         */
        void copy(const ASISTStudy3InterventionEstimator& estimator);

        //------------------------------------------------------------------
        // Data members
        //------------------------------------------------------------------

      private:
        //------------------------------------------------------------------
        // Types & Consts
        //------------------------------------------------------------------
        const static int MAX_DIST = 5;
        const static int VICINITY_MAX_RADIUS = 5;

        //------------------------------------------------------------------
        // Static functions
        //------------------------------------------------------------------
        static bool did_player_speak_about_marker(
            int player_order,
            const ASISTStudy3MessageConverter::Marker& unspoken_marker,
            int time_step,
            const EvidenceSet& new_data);

        static ASISTStudy3MessageConverter::Marker get_last_placed_marker(
            int player_order, int time_step, const EvidenceSet& new_data);

        static bool did_player_interact_with_victim(
            int player_order, int time_step, const EvidenceSet& new_data);

        static bool is_player_far_apart(
            int player_order,
            const ASISTStudy3MessageConverter::Position& position,
            int max_distance,
            int time_step,
            const EvidenceSet& new_data);

        static std::vector<ASISTStudy3MessageConverter::Marker>
        get_removed_markers(int player_order,
                            int time_step,
                            const EvidenceSet& new_data);

        static bool did_player_change_area(int player_order,
                                           int time_step,
                                           const EvidenceSet& new_data);

        //------------------------------------------------------------------
        // Member functions
        //------------------------------------------------------------------

        /**
         * Initialize containers with the number of trials being estimated.
         *
         * @param new_data: evidence
         */
        void initialize_containers(const EvidenceSet& new_data);

        /**
         * Estimate if team is motivated.
         *
         * @param new_data: evidence
         */
        void estimate_motivation(const EvidenceSet& new_data);

        /**
         * Estimate if players placed markers and did not talk about it.
         *
         * @param new_data: evidence
         */
        void estimate_unspoken_markers(const EvidenceSet& new_data);

        //------------------------------------------------------------------
        // Data members
        //------------------------------------------------------------------

        bool containers_initialized = false;

        int last_time_step = -1;
        bool first_mission = true;
        double encouragement_cdf = 0;

        std::vector<ASISTStudy3MessageConverter::Marker> last_placed_markers;
        std::vector<ASISTStudy3MessageConverter::Marker>
            active_unspoken_markers;
    };

} // namespace tomcat::model
