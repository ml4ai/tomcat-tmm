#pragma once

#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

#include "asist/study3/ASISTStudy3InterventionLogger.h"
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
        //------------------------------------------------------------------
        // Types & Consts
        //------------------------------------------------------------------
        inline static const int VICINITY_MAX_RADIUS = 5;
        inline static const int HELP_REQUEST_LATENCY = 10;
        inline static const int HELP_REQUEST_REPLY_LATENCY = 10;

        inline static const std::string NAME =
            "asist_study3_intervention_estimator";

        //------------------------------------------------------------------
        // Constructors & Destructor
        //------------------------------------------------------------------

        ASISTStudy3InterventionEstimator(const ModelPtr& model);

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

        void set_logger(const OnlineLoggerPtr& logger) override;

        /**
         * Get the CDF of the number of encouragement utterances identified in
         * mission 1.
         *
         * @return CDF
         */
        double get_encouragement_cdf() const;

        /**
         * Get the number of encouragement utterances identified so far.
         *
         * @return Number of encouragement utterances
         */
        int get_num_encouragements() const;

        bool
        is_help_request_critical_victim_intervention_active(int player_order);

        bool is_help_request_room_escape_intervention_active(int player_order);

        bool is_help_request_reply_intervention_active(int player_order);

        /**
         * Removes current active unspoken marker from the list.
         *
         * @param player_order: index of the player that has an active unspoken
         * marker
         */
        void clear_active_unspoken_marker(int player_order);

        /**
         * Inactivates ask-for-help intervention
         *
         * @param player_order: index of the player that has an active unspoken
         * marker
         */
        void
        restart_help_request_critical_victim_intervention(int player_order);

        void restart_help_request_room_escape_intervention(int player_order);

        void restart_help_request_reply_intervention(int player_order);

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
        // Types
        //------------------------------------------------------------------

        enum InterventionState { WATCHED, ACTIVE, NONE };

        //------------------------------------------------------------------
        // Static functions
        //------------------------------------------------------------------
        static bool did_player_speak_about_marker(
            int player_order,
            const ASISTStudy3MessageConverter::Marker& unspoken_marker,
            int time_step,
            const EvidenceSet& new_data);

        static std::unordered_set<ASISTStudy3MessageConverter::MarkerType>
        get_mentioned_marker_types(int player_order,
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

        static bool did_player_remove_marker(
            const ASISTStudy3MessageConverter::Marker& marker,
            int player_order,
            int time_step,
            const EvidenceSet& new_data);

        static bool does_player_need_help_to_wake_victim(
            int player_order, int time_step, const EvidenceSet& new_data);

        static bool did_player_ask_for_help(int player_order,
                                            int time_step,
                                            const EvidenceSet& new_data);

        static bool is_there_another_player_in_same_area(
            int player_order, int time_step, const EvidenceSet& new_data);

        static bool did_player_speak_about_critical_victim(
            int player_order, int time_step, const EvidenceSet& new_data);

        static bool does_player_need_help_to_exit_room(
            int player_order, int time_step, const EvidenceSet& new_data);

        static std::string get_active_threat_id(int player_order,
                                                int time_step,
                                                const EvidenceSet& new_data);

        static bool is_player_inside_room(int player_order,
                                          int time_step,
                                          const EvidenceSet& new_data);

        static bool is_engineer_in_same_room(int player_order,
                                             int time_step,
                                             const EvidenceSet& new_data);

        static bool should_watch_marker_type(
            const ASISTStudy3MessageConverter::MarkerType& marker_type);

        static int get_helper_player_order(int assisted_player_order,
                                           int time_step,
                                           const EvidenceSet& new_data);

        static bool is_player_being_released(int player_order,
                                             int time_step,
                                             const EvidenceSet& new_data,
                                             const std::string& threat_id);

        //------------------------------------------------------------------
        // Member functions
        //------------------------------------------------------------------

        /**
         * Initialize containers with the number of trials being estimated.
         *
         * @param new_data: evidence
         */
        void initialize_containers(const EvidenceSet& new_data);

        void update_communication(int player_order,
                                  int time_step,
                                  const EvidenceSet& new_data);

        /**
         * Estimate if team is motivated.
         *
         * @param new_data: evidence
         */
        void estimate_motivation(int time_step, const EvidenceSet& new_data);

        /**
         * Estimate if players placed markers and did not talk about it.
         *
         * @param new_data: evidence
         */
        void estimate_communication_marker(int player_order,
                                           int time_step,
                                           const EvidenceSet& new_data);

        /**
         * Estimate if players ask for help when they need it.
         *
         * @param new_data: evidence
         */
        void estimate_help_request(int player_order,
                                   int time_step,
                                   const EvidenceSet& new_data);

        void estimate_help_request_critical_victim(int player_order,
                                                   int time_step,
                                                   const EvidenceSet& new_data);

        void estimate_help_request_room_escape(int player_order,
                                               int time_step,
                                               const EvidenceSet& new_data);

        void estimate_help_request_reply(int player_order,
                                      int time_step,
                                      const EvidenceSet& new_data);

        //------------------------------------------------------------------
        // Data members
        //------------------------------------------------------------------

        std::shared_ptr<ASISTStudy3InterventionLogger> custom_logger;

        bool containers_initialized = false;

        int last_time_step = -1;
        bool first_mission = true;

        std::vector<ASISTStudy3MessageConverter::Marker> watched_markers;
        std::vector<ASISTStudy3MessageConverter::Marker> active_markers;

        // Variables to keep track of the state machine
        std::vector<InterventionState> help_request_critical_victim_state;
        std::vector<int> help_request_critical_victim_timer;
        std::vector<InterventionState> help_request_room_escape_state;
        std::vector<int> help_request_room_escape_timer;
        std::vector<InterventionState> help_request_reply_state;
        std::vector<int> help_request_reply_timer;

        // Variables to keep track of information that needs to persist beyond a
        // time step
        std::vector<std::unordered_set<ASISTStudy3MessageConverter::MarkerType>>
            mentioned_marker_types;
        std::vector<bool> recently_mentioned_critical_victim;
        std::vector<bool> recently_mentioned_help_request;
        std::vector<std::string> latest_active_threat_id;
    };

} // namespace tomcat::model
