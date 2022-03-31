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
         * @return CDFs. One per trial if multiple trials are being processed at
         * the same time.
         */
        Eigen::VectorXd get_encouragement_cdfs();

        /**
         * Removes current active unspoken marker from the list.
         *
         * @param player_order: index of the player that has an active unspoken
         * marker
         * @param data_point: trial data index
         */
        void clear_active_unspoken_marker(int player_order, int data_point);

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

        const std::vector<std::vector<Marker>>&
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

        //------------------------------------------------------------------
        // Structs
        //------------------------------------------------------------------
        struct Position {
            double x;
            double z;

            Position() : x(0), z(0) {}

            Position(double x, double z) : x(x), z(z) {}

            double distance_to(const Position& pos) const {
                return sqrt(this->x * pos.x + this->z * pos.z);
            }
        };

        struct Marker {
            ASISTStudy3MessageConverter::MarkerType type;
            Position position;

            Marker()
                : type(ASISTStudy3MessageConverter::MarkerType::NONE),
                  position(Position()) {}

            Marker(ASISTStudy3MessageConverter::MarkerType type,
                   const Position& position)
                : type(type), position(position) {}

            bool is_none() const {
                return this->type ==
                       ASISTStudy3MessageConverter::MarkerType::NONE;
            }
        };

        //------------------------------------------------------------------
        // Static functions
        //------------------------------------------------------------------
        static bool did_player_speak_about_marker(int player_order,
                                                  const Marker& unspoken_marker,
                                                  int data_point,
                                                  int time_step,
                                                  const EvidenceSet& new_data);

        static Marker get_last_placed_marker(int player_order,
                                             int data_point,
                                             int time_step,
                                             const EvidenceSet& new_data);

        static bool
        did_player_interact_with_victim(int player_order,
                                        int data_point,
                                        int time_step,
                                        const EvidenceSet& new_data);

        static bool is_player_far_apart(int player_order,
                                        Position position,
                                        int max_distance,
                                        int data_point,
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

        /**
         * Checks if a player changed area in the map.
         *
         * @param player_order: player's index
         * @param data_point: trial data index
         * @param time_step: time step
         * @param new_data: evidence
         * @return
         */
        bool did_player_change_area(int player_order,
                                    int data_point,
                                    int time_step,
                                    const EvidenceSet& new_data);

        //------------------------------------------------------------------
        // Data members
        //------------------------------------------------------------------

        bool containers_initialized = false;

        int last_time_step = -1;
        bool first_mission = true;
        Eigen::VectorXd encouragement_cdf = Eigen::VectorXd(0);

        std::vector<std::vector<std::string>> last_areas;
        std::vector<std::vector<Marker>> last_placed_markers;
        std::vector<std::vector<Marker>> active_unspoken_markers;

        //        std::shared_ptr<ASISTStudy3InterventionModel>
        //        intervention_model; int num_encouragements_first_mission = 0;
        //
        //        // Indices of each role
        //        const static int NUM_ROLES = 3;
        //
        //        const static int MEDIC = 0;
        //        const static int TRANSPORTER = 1;
        //        const static int ENGINEER = 2;
        //
        //        std::unordered_map<std::string, int> room_id_to_idx;
        //        std::vector<std::string> room_ids;
        //
        //        // Stores belief estimates about threat rooms per player and
        //        room. std::vector<std::vector<SumProductEstimator>>
        //            threat_room_belief_estimators;
    };

} // namespace tomcat::model
