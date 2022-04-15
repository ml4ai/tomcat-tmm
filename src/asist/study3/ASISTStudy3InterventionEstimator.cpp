#include "ASISTStudy3InterventionEstimator.h"

#include <algorithm>

#include <fmt/format.h>

#include "asist/study3/ASISTStudy3InterventionModel.h"
#include "asist/study3/ASISTStudy3MessageConverter.h"
#include "utils/EigenExtensions.h"
#include "utils/JSONChecker.h"

namespace tomcat::model {

    using namespace std;
    using Labels = ASISTStudy3MessageConverter::Labels;
    using MarkerType = ASISTStudy3MessageConverter::MarkerType;
    using Marker = ASISTStudy3MessageConverter::Marker;
    using Position = ASISTStudy3MessageConverter::Position;

    //----------------------------------------------------------------------
    // Constructors
    //----------------------------------------------------------------------
    ASISTStudy3InterventionEstimator::ASISTStudy3InterventionEstimator(
        const ModelPtr& model)
        : Estimator(model) {

        if (!dynamic_pointer_cast<ASISTStudy3InterventionModel>(model)) {
            throw TomcatModelException(
                fmt::format("ASISTStudy3InterventionEstimator requires "
                            "an ASISTStudy3InterventionModel."));
        }
    }

    //----------------------------------------------------------------------
    // Copy & Move constructors/assignments
    //----------------------------------------------------------------------
    ASISTStudy3InterventionEstimator::ASISTStudy3InterventionEstimator(
        const ASISTStudy3InterventionEstimator& estimator) {
        this->copy(estimator);
    }

    ASISTStudy3InterventionEstimator&
    ASISTStudy3InterventionEstimator::operator=(
        const ASISTStudy3InterventionEstimator& estimator) {
        this->copy(estimator);
        return *this;
    }

    //----------------------------------------------------------------------
    // Static functions
    //----------------------------------------------------------------------
    bool ASISTStudy3InterventionEstimator::did_player_speak_about_marker(
        int player_order,
        const Marker& unspoken_marker,
        int time_step,
        const EvidenceSet& new_data) {

        const auto& json_dialog =
            new_data.get_dict_like_data()[0][time_step][Labels::DIALOG]
                                         [player_order];

        return EXISTS(
            unspoken_marker.type,
            get_mentioned_marker_types(player_order, time_step, new_data));
    }

    unordered_set<MarkerType>
    ASISTStudy3InterventionEstimator::get_mentioned_marker_types(
        int player_order, int time_step, const EvidenceSet& new_data) {

        const auto& json_dialog =
            new_data.get_dict_like_data()[0][time_step][Labels::DIALOG]
                                         [player_order];

        unordered_set<MarkerType> marker_types;

        if ((bool)json_dialog["no_victim"]) {
            marker_types.insert(MarkerType::NO_VICTIM);
        }
        else if ((bool)json_dialog["regular_victim"]) {
            marker_types.insert(MarkerType::REGULAR_VICTIM);
        }
        else if ((bool)json_dialog["critical_victim"]) {
            marker_types.insert(MarkerType::VICTIM_C);
        }
        else if ((bool)json_dialog["victim_a"]) {
            marker_types.insert(MarkerType::VICTIM_A);
        }
        else if ((bool)json_dialog["victim_b"]) {
            marker_types.insert(MarkerType::VICTIM_B);
        }
        else if ((bool)json_dialog["obstacle"]) {
            marker_types.insert(MarkerType::RUBBLE);
        }
        else if ((bool)json_dialog["threat"]) {
            marker_types.insert(MarkerType::THREAT_ROOM);
        }
        else if ((bool)json_dialog["help_needed"]) {
            marker_types.insert(MarkerType::SOS);
        }

        return marker_types;
    }

    Marker ASISTStudy3InterventionEstimator::get_last_placed_marker(
        int player_order, int time_step, const EvidenceSet& new_data) {

        const auto& json_marker =
            new_data
                .get_dict_like_data()[0][time_step][Labels::LAST_PLACED_MARKERS]
                                     [player_order];

        if (!json_marker.empty()) {
            return Marker(json_marker);
        }

        return Marker();
    }

    bool ASISTStudy3InterventionEstimator::did_player_interact_with_victim(
        int player_order, int time_step, const EvidenceSet& new_data) {

        return new_data
            .get_dict_like_data()[0][time_step][Labels::VICTIM_INTERACTIONS]
                                 [player_order];
    }

    bool ASISTStudy3InterventionEstimator::is_player_far_apart(
        int player_order,
        const Position& position,
        int max_distance,
        int time_step,
        const EvidenceSet& new_data) {

        Position player_pos(
            new_data
                .get_dict_like_data()[0][time_step][Labels::PLAYER_POSITIONS]
                                     [player_order]);
        return player_pos.distance_to(position) > max_distance;
    }

    vector<Marker> ASISTStudy3InterventionEstimator::get_removed_markers(
        int player_order, int time_step, const EvidenceSet& new_data) {

        vector<Marker> removed_markers;
        const auto& json_markers =
            new_data.get_dict_like_data()[0][time_step][Labels::REMOVED_MARKERS]
                                         [player_order];
        for (const auto& json_marker : json_markers) {
            removed_markers.push_back(Marker(json_marker));
        }

        return removed_markers;
    }

    bool ASISTStudy3InterventionEstimator::did_player_change_area(
        int player_order, int time_step, const EvidenceSet& new_data) {

        return new_data
            .get_dict_like_data()[0][time_step][Labels::LOCATION_CHANGES]
                                 [player_order];
    }

    bool ASISTStudy3InterventionEstimator::did_player_remove_marker(
        const ASISTStudy3MessageConverter::Marker& marker,
        int player_order,
        int time_step,
        const EvidenceSet& new_data) {

        const auto& removed_markers =
            get_removed_markers(player_order, time_step, new_data);

        return any_of(removed_markers.begin(),
                      removed_markers.end(),
                      [&m = marker](const Marker& removed_marker) {
                          return removed_marker == m;
                      });
    }

    bool ASISTStudy3InterventionEstimator::does_player_need_help_to_wake_victim(
        int player_order, int time_step, const EvidenceSet& new_data) {

        double distance =
            new_data
                .get_dict_like_data()[0][time_step][Labels::FOV][player_order]
                                     ["distance_to_critical_victim"];

        // If there's another player in the same area, there's no need to
        // ask for help.
        return distance < VICINITY_MAX_RADIUS;
    }

    bool ASISTStudy3InterventionEstimator::did_player_ask_for_help(
        int player_order, int time_step, const EvidenceSet& new_data) {

        const auto& json_dialog =
            new_data.get_dict_like_data()[0][time_step][Labels::DIALOG]
                                         [player_order];

        return json_dialog["help_needed"];
    }

    bool ASISTStudy3InterventionEstimator::is_there_another_player_around(
        int player_order, int time_step, const EvidenceSet& new_data) {

        const string& player_location =
            new_data.get_dict_like_data()[0][time_step][Labels::LOCATIONS]
                                         [player_order]["id"];
        bool around = false;
        for (int i = 0; i < 3; i++) {
            if (i != player_order) {
                const string& other_player_location =
                    new_data
                        .get_dict_like_data()[0][time_step][Labels::LOCATIONS]
                                             [player_order]["id"];

                around = player_location == other_player_location;
            }
        }

        return around;
    }

    bool
    ASISTStudy3InterventionEstimator::did_player_speak_about_critical_victim(
        int player_order, int time_step, const EvidenceSet& new_data) {

        const auto& json_dialog =
            new_data.get_dict_like_data()[0][time_step][Labels::DIALOG]
                                         [player_order];

        return json_dialog["critical_victim"];
    }

    bool ASISTStudy3InterventionEstimator::does_player_need_help_to_exit_room(
        int player_order, int time_step, const EvidenceSet& new_data) {

        return !get_threat_id(player_order, time_step, new_data).empty() &&
               is_player_in_room(player_order, time_step, new_data);
    }

    string ASISTStudy3InterventionEstimator::get_threat_id(
        int player_order, int time_step, const EvidenceSet& new_data) {

        string threat_id =
            new_data.get_dict_like_data()[0][time_step][Labels::FOV]
                                         [player_order]["collapsed_rubble_id"];
        return threat_id;
    }

    bool ASISTStudy3InterventionEstimator::is_player_in_room(
        int player_order, int time_step, const EvidenceSet& new_data) {

        return new_data.get_dict_like_data()[0][time_step][Labels::LOCATIONS]
                                            [player_order]["room"];
    }

    bool ASISTStudy3InterventionEstimator::is_engineer_around(
        int player_order, int time_step, const EvidenceSet& new_data) {

        int engineer_order = new_data.get_metadata()[0]["engineer_order"];

        bool around;
        if (player_order == engineer_order) {
            around = true;
        }
        else {
            // Check if the engineer is in the same location as the player
            const string& player_location =
                new_data.get_dict_like_data()[0][time_step][Labels::LOCATIONS]
                                             [player_order]["id"];

            const string& engineer_location =
                new_data.get_dict_like_data()[0][time_step][Labels::LOCATIONS]
                                             [engineer_order]["id"];

            around = player_location == engineer_location;
        }

        return around;
    }

    bool should_watch_marker_type(
        const ASISTStudy3MessageConverter::MarkerType& marker_type) {
        return marker_type == MarkerType::VICTIM_C ||
               marker_type == MarkerType::REGULAR_VICTIM ||
               marker_type == MarkerType::RUBBLE ||
               marker_type == MarkerType::SOS;
    }

    int ASISTStudy3InterventionEstimator::get_helper_player_order(
        int assisted_player_order, int time_step, const EvidenceSet& new_data) {

        for (int helper_player_order = 0; helper_player_order < 3;
             helper_player_order++) {
            if (helper_player_order != assisted_player_order) {
                const auto& json_dialog =
                    new_data.get_dict_like_data()[0][time_step][Labels::DIALOG]
                                                 [helper_player_order];

                if ((bool)json_dialog["help_on_the_way"]) {
                    return helper_player_order;
                }
            }
        }

        return -1;
    }

    bool ASISTStudy3InterventionEstimator::is_player_being_released(
        int player_order,
        int time_step,
        const EvidenceSet& new_data,
        const string& threat_id) {

        const string& threat_id_being_removed =
            new_data.get_dict_like_data()
                [0][time_step][Labels::RUBBLE_COLLAPSE]
                ["destruction_interaction_collapsed_rubble_id"];

        if (threat_id.empty() || threat_id_being_removed.empty()) {
            return false;
        }

        return threat_id == threat_id_being_removed;
    }

    //----------------------------------------------------------------------
    // Member functions
    //----------------------------------------------------------------------
    void ASISTStudy3InterventionEstimator::copy(
        const ASISTStudy3InterventionEstimator& estimator) {
        Estimator::copy(estimator);

        this->last_time_step = estimator.last_time_step;
        // TODO
    }

    void
    ASISTStudy3InterventionEstimator::estimate(const EvidenceSet& new_data) {

        this->initialize_containers(new_data);

        this->first_mission = new_data.get_metadata()[0]["mission_order"] == 1;

        for (int t = 0; t < new_data.get_time_steps(); t++) {
            this->estimate_motivation(t, new_data);

            for (int player_order = 0; player_order < 3; player_order++) {
                this->update_communication(player_order, t, new_data);
                this->estimate_communication_marker(player_order, t, new_data);
                this->estimate_help_request(player_order, t, new_data);
                this->estimate_help_on_the_way(player_order, t, new_data);
            }
        }

        this->last_time_step += new_data.get_time_steps();
    }

    void ASISTStudy3InterventionEstimator::initialize_containers(
        const EvidenceSet& new_data) {
        if (!this->containers_initialized) {
            this->watched_markers = vector<Marker>(3);
            this->watched_critical_victims = vector<int>(3, -1);
            this->watched_threats = vector<pair<string, int>>(3, {"", -1});
            this->watched_no_help_on_the_way = vector<int>(3, -1);
            this->active_markers = vector<Marker>(3);
            this->active_no_critical_victim_help_requests =
                vector<bool>(3, false);
            this->active_no_threat_help_requests = vector<bool>(3, false);
            this->active_no_help_on_the_way = vector<bool>(3, false);
            this->mentioned_marker_types = vector<unordered_set<MarkerType>>(3);
            this->mentioned_critical_victim = vector<bool>(3, false);
            this->mentioned_help_request = vector<bool>(3, false);

            this->help_request_room_escape_state =
                vector<InterventionState>(3, InterventionState::NONE);
            this->help_request_room_escape_timer = vector<int>(3, 0);
            this->help_request_room_escape_watched_threat_ids =
                vector<string>(3, "");

            this->containers_initialized = true;
        }
    }

    void ASISTStudy3InterventionEstimator::update_communication(
        int player_order, int time_step, const EvidenceSet& new_data) {

        if (did_player_change_area(player_order, time_step, new_data)) {
            // Recent mention to tracked information is cleared when the
            // player changes location.
            this->mentioned_marker_types[player_order].clear();
            this->mentioned_critical_victim[player_order] = false;
            this->mentioned_help_request[player_order] = false;
        }

        const auto& marker_types =
            get_mentioned_marker_types(player_order, time_step, new_data);

        this->mentioned_marker_types[player_order].insert(marker_types.begin(),
                                                          marker_types.end());

        const auto& json_dialog =
            new_data.get_dict_like_data()[0][time_step][Labels::DIALOG]
                                         [player_order];

        if ((bool)json_dialog["critical_victim"]) {
            this->mentioned_critical_victim[player_order] = true;
        }
        else if ((bool)json_dialog["help_needed"]) {
            this->mentioned_help_request[player_order] = true;
        }
    }

    void ASISTStudy3InterventionEstimator::estimate_motivation(
        int time_step, const EvidenceSet& new_data) {
        const auto& metadata = new_data.get_metadata();

        check_field(metadata[0], "mission_order");

        auto& encouragement_node =
            dynamic_pointer_cast<ASISTStudy3InterventionModel>(this->model)
                ->get_encouragement_node();

        int increments = 0;
        if (metadata[0]["mission_order"] == 1) {
            if (this->last_time_step < 0) {
                encouragement_node->set_assignment(Eigen::VectorXd::Zero(1));
                this->custom_logger->log_watch_motivation_intervention(0);
            }

            increments +=
                (int)new_data
                    .get_dict_like_data()[0][time_step][Labels::ENCOURAGEMENT];

            encouragement_node->increment_assignment(increments);

            if (increments > 0) {
                this->custom_logger->log_update_motivation_intervention(
                    this->last_time_step + new_data.get_time_steps(),
                    increments);
            }
        }
        else {
            if (encouragement_node->get_size() == 0) {
                this->custom_logger->log_empty_encouragements(
                    this->last_time_step + new_data.get_time_steps());
                // The agent crashed and data from mission 1 was lost.
                // Assume there was no encouragement utterances to force
                // intervention.
                encouragement_node->set_assignment(Eigen::VectorXd::Zero(1));
            }
        }
    }

    void ASISTStudy3InterventionEstimator::estimate_communication_marker(
        int player_order, int time_step, const EvidenceSet& new_data) {

        if (!this->watched_markers[player_order].is_none()) {
            // Cancel watched marker if possible
            bool marker_removal =
                did_player_remove_marker(this->watched_markers[player_order],
                                         player_order,
                                         time_step,
                                         new_data);
            bool speech = did_player_speak_about_marker(
                player_order,
                this->watched_markers[player_order],
                time_step,
                new_data);

            if (marker_removal || speech) {
                this->custom_logger
                    ->log_cancel_communication_marker_intervention(
                        this->last_time_step + time_step + 1,
                        player_order,
                        this->watched_markers[player_order],
                        speech,
                        marker_removal);
                this->watched_markers[player_order] = Marker();
            }
        }

        Marker new_marker =
            get_last_placed_marker(player_order, time_step, new_data);
        bool area_changed =
            did_player_change_area(player_order, time_step, new_data);
        bool victim_interaction =
            did_player_interact_with_victim(player_order, time_step, new_data);
        bool marker_placed = !new_marker.is_none();
        bool count_as_new_marker = true;
        if (!this->watched_markers[player_order].is_none() && marker_placed) {
            double distance = new_marker.position.distance_to(
                this->watched_markers[player_order].position);
            if (distance <= VICINITY_MAX_RADIUS) {
                // If the new marker is too close from previous
                // placed markers, don't count it as a new marker.
                count_as_new_marker = false;
            }
        }

        if (EXISTS(new_marker.type,
                   this->mentioned_marker_types[player_order])) {
            // The player spoke about the new marker before placing
            // it. Therefore, we don't have to intervene.
            count_as_new_marker = false;

            this->custom_logger->log_hinder_communication_marker_intervention(
                this->last_time_step + time_step + 1, player_order, new_marker);
        }

        bool activate_intervention =
            !this->watched_markers[player_order].is_none() &&
            (area_changed || victim_interaction ||
             (marker_placed && count_as_new_marker));

        if (activate_intervention) {
            this->active_markers[player_order] =
                this->watched_markers[player_order];

            this->custom_logger->log_activate_communication_marker_intervention(
                this->last_time_step + time_step + 1,
                player_order,
                this->watched_markers[player_order],
                area_changed,
                victim_interaction,
                marker_placed);

            this->watched_markers[player_order] = Marker();
        }

        if (marker_placed && count_as_new_marker &&
            tomcat::model::should_watch_marker_type(new_marker.type)) {
            // Watch the new marker for intervention
            this->watched_markers[player_order] = new_marker;

            this->custom_logger->log_watch_communication_marker_intervention(
                this->last_time_step + time_step + 1, player_order, new_marker);
        }
    }

    void ASISTStudy3InterventionEstimator::estimate_help_request(
        int player_order, int time_step, const EvidenceSet& new_data) {
        this->estimate_critical_victim_help_request(
            player_order, time_step, new_data);
        this->estimate_threat_help_request(player_order, time_step, new_data);
    }

    void
    ASISTStudy3InterventionEstimator::estimate_critical_victim_help_request(
        int player_order, int time_step, const EvidenceSet& new_data) {

        if (this->watched_critical_victims[player_order] >= 0) {
            // Cancel intervention if possible
            bool area_changed =
                did_player_change_area(player_order, time_step, new_data);
            bool help_requested =
                did_player_ask_for_help(player_order, time_step, new_data);
            bool mention_to_critical_victim =
                did_player_speak_about_critical_victim(
                    player_order, time_step, new_data);
            int other_players_around = is_there_another_player_around(
                player_order, time_step, new_data);
            if (area_changed || help_requested || mention_to_critical_victim ||
                other_players_around) {
                this->active_no_critical_victim_help_requests[player_order] =
                    false;
                this->watched_critical_victims[player_order] = -1;

                this->custom_logger
                    ->log_cancel_ask_for_help_critical_victim_intervention(
                        this->last_time_step + time_step + 1,
                        player_order,
                        area_changed,
                        help_requested,
                        mention_to_critical_victim,
                        other_players_around);
            }
        }

        bool recent_mention_to_critical_victim =
            this->mentioned_critical_victim[player_order];
        bool activate_intervention =
            does_player_need_help_to_wake_victim(
                player_order, time_step, new_data) &&
            !is_there_another_player_around(player_order, time_step, new_data);

        // Watch and activate intervention
        if (activate_intervention && !recent_mention_to_critical_victim) {
            this->watched_critical_victims[player_order] =
                this->last_time_step + time_step + 1;

            this->custom_logger
                ->log_watch_ask_for_help_critical_victim_intervention(
                    this->last_time_step + time_step + 1, player_order);
        }
        else {
            if (activate_intervention && recent_mention_to_critical_victim) {
                this->custom_logger
                    ->log_hinder_ask_for_help_critical_victim_intervention(
                        this->last_time_step + time_step + 1, player_order);
            }

            if (this->watched_critical_victims[player_order] >= 0) {
                // We are already watching this player for help
                // request regarding critical victim awakening
                if (this->last_time_step + time_step + 1 -
                        this->watched_critical_victims[player_order] >
                    ASK_FOR_HELP_LATENCY) {
                    // It has passed enough time. Restart watching
                    // time and activate intervention.
                    this->watched_critical_victims[player_order] =
                        this->last_time_step + time_step + 1;
                    this->active_no_critical_victim_help_requests
                        [player_order] = true;

                    this->custom_logger
                        ->log_activate_ask_for_help_critical_victim_intervention(
                            this->last_time_step + time_step + 1,
                            player_order,
                            ASK_FOR_HELP_LATENCY);
                }
            }
        }
    }

    void ASISTStudy3InterventionEstimator::estimate_threat_help_request(
        int player_order, int time_step, const EvidenceSet& new_data) {

        if (this->help_request_room_escape_state[player_order] ==
            InterventionState::NONE) {
            bool is_trapped = does_player_need_help_to_exit_room(
                player_order, time_step, new_data);
            bool is_engineer_in_room =
                is_engineer_around(player_order, time_step, new_data);

            if (is_trapped && !is_engineer_in_room) {
                string threat_id =
                    get_threat_id(player_order, time_step, new_data);
                bool is_being_released = is_player_being_released(
                    player_order, time_step, new_data, threat_id);
                bool recent_mention_to_help =
                    this->mentioned_help_request[player_order];

                if (recent_mention_to_help || is_being_released) {
                    this->custom_logger
                        ->log_hinder_ask_for_help_threat_intervention(
                            this->last_time_step + time_step + 1,
                            player_order,
                            recent_mention_to_help,
                            is_being_released);
                }
                else {
                    this->help_request_room_escape_state[player_order] =
                        InterventionState::WATCHED;
                    this->help_request_room_escape_timer[player_order] =
                        ASK_FOR_HELP_LATENCY;
                    this->help_request_room_escape_watched_threat_ids
                        [player_order] =
                        get_threat_id(player_order, time_step, new_data);

                    this->custom_logger
                        ->log_watch_ask_for_help_threat_intervention(
                            this->last_time_step + time_step + 1, player_order);
                }
            }
        }
        else {
            // Check if it can be canceled
            bool help_requested =
                did_player_ask_for_help(player_order, time_step, new_data);
            bool left_room =
                !is_player_in_room(player_order, time_step, new_data);
            bool is_engineer_in_room =
                is_engineer_around(player_order, time_step, new_data);
            bool being_released = this->is_player_being_released(
                player_order,
                time_step,
                new_data,
                this->help_request_room_escape_watched_threat_ids
                    [player_order]);

            if (help_requested || left_room || is_engineer_in_room ||
                being_released) {
                this->help_request_room_escape_state[player_order] =
                    InterventionState::NONE;

                this->custom_logger
                    ->log_cancel_ask_for_help_threat_intervention(
                        this->last_time_step + time_step + 1,
                        player_order,
                        left_room,
                        help_requested,
                        being_released,
                        is_engineer_in_room);
            }

            if (this->help_request_room_escape_state[player_order] ==
                InterventionState::WATCHED) {
                // Decrement counter
                this->help_request_room_escape_timer[player_order] -= 1;
                if (this->help_request_room_escape_timer[player_order] == 0) {
                    this->help_request_room_escape_state[player_order] =
                        InterventionState::ACTIVE;

                    this->custom_logger
                        ->log_activate_ask_for_help_threat_intervention(
                            this->last_time_step + time_step + 1,
                            player_order,
                            ASK_FOR_HELP_LATENCY);
                }
            }
        }

        //        if (!this->watched_threats[player_order].first.empty()) {
        //            // Cancel intervention if possible
        //            bool help_requested =
        //                did_player_ask_for_help(player_order, time_step,
        //                new_data);
        //            if (!is_in_room || help_requested || being_released) {
        //                this->active_no_threat_help_requests[player_order] =
        //                false; this->watched_threats[player_order] = {"", -1};
        //
        //                this->custom_logger
        //                    ->log_cancel_ask_for_help_threat_intervention(
        //                        this->last_time_step + time_step + 1,
        //                        player_order,
        //                        !is_in_room,
        //                        help_requested,
        //                        being_released);
        //            }
        //        }
        //
        //        // Watch and activate
        //        bool is_trapped = does_player_need_help_to_exit_room(
        //            player_order, time_step, new_data);
        //        bool is_engineer_nearby =
        //            is_engineer_around(player_order, time_step, new_data);
        //        bool recent_mention_to_help =
        //            this->mentioned_help_request[player_order];
        //        bool activate_intervention =
        //            !being_released && is_trapped && !is_engineer_nearby &&
        //            is_in_room;
        //
        //        if (activate_intervention && !recent_mention_to_help) {
        //            string threat_id = get_threat_id(player_order, time_step,
        //            new_data);
        //
        //            this->watched_threats[player_order] = {
        //                threat_id, this->last_time_step + time_step + 1};
        //
        //            this->custom_logger->log_watch_ask_for_help_threat_intervention(
        //                this->last_time_step + time_step + 1, player_order);
        //        }
        //        else {
        //            if (activate_intervention && recent_mention_to_help) {
        //                this->custom_logger
        //                    ->log_hinder_ask_for_help_threat_intervention(
        //                        this->last_time_step + time_step + 1,
        //                        player_order);
        //            }
        //
        //            if (!this->watched_threats[player_order].first.empty()) {
        //                // We are already watching this player for help
        //                // request regarding exiting a threat room
        //                if (this->last_time_step + time_step + 1 -
        //                        this->watched_threats[player_order].second >
        //                    ASK_FOR_HELP_LATENCY) {
        //                    // It has passed enough time. Restart watching
        //                    // time and activate intervention.
        //                    string threat_id =
        //                        get_threat_id(player_order, time_step,
        //                        new_data);
        //                    this->watched_threats[player_order] = {
        //                        threat_id, this->last_time_step + time_step +
        //                        1};
        //                    this->active_no_threat_help_requests[player_order]
        //                    = true;
        //
        //                    this->custom_logger
        //                        ->log_activate_ask_for_help_threat_intervention(
        //                            this->last_time_step + time_step + 1,
        //                            player_order,
        //                            ASK_FOR_HELP_LATENCY);
        //                }
        //            }
        //        }
    }

    void ASISTStudy3InterventionEstimator::estimate_help_on_the_way(
        int player_order, int time_step, const EvidenceSet& new_data) {

        bool changed_area =
            did_player_change_area(player_order, time_step, new_data);
        int helper_player_order =
            get_helper_player_order(player_order, time_step, new_data);
        bool help_request_answered = helper_player_order >= 0;

        if (this->watched_no_help_on_the_way[player_order] >= 0) {
            // Cancel intervention if possible
            if (changed_area || help_request_answered) {
                this->active_no_help_on_the_way[player_order] = false;
                this->watched_no_help_on_the_way[player_order] = -1;

                this->custom_logger->log_cancel_help_on_the_way_intervention(
                    this->last_time_step + time_step + 1,
                    player_order,
                    helper_player_order,
                    changed_area,
                    help_request_answered);
            }
        }

        // Watch and activate
        bool asked_for_help =
            did_player_ask_for_help(player_order, time_step, new_data);
        if (asked_for_help) {
            this->watched_no_help_on_the_way[player_order] =
                this->last_time_step + time_step + 1;

            this->custom_logger->log_watch_help_on_the_way_intervention(
                this->last_time_step + time_step + 1, player_order);
        }
        else if (this->watched_no_help_on_the_way[player_order] >= 0) {
            // We are already watching this player for help
            // request answer
            if (this->last_time_step + time_step + 1 -
                    this->watched_no_help_on_the_way[player_order] >
                ASK_FOR_HELP_LATENCY) {

                // It has passed enough time. Restart watching
                // time and activate intervention.
                this->watched_no_help_on_the_way[player_order] =
                    this->last_time_step + time_step + 1;
                this->active_no_help_on_the_way[player_order] = true;

                this->custom_logger->log_activate_help_on_the_way_intervention(
                    this->last_time_step + time_step + 1,
                    player_order,
                    ASK_FOR_HELP_LATENCY);
            }
        }
    }

    void
    ASISTStudy3InterventionEstimator::get_info(nlohmann::json& json) const {
        // TODO - maybe it is not necessary
        //            nlohmann::json json_info;
        //            json_info["name"] = this->get_name();
        //            if (this->first_mission) {
        //                json_info["num_encouragements"] = to_string(
        //                    dynamic_pointer_cast<ASISTStudy3InterventionModel>(
        //                        this->model)
        //                        ->get_encouragement_node()
        //                        ->get_assignment());
        //            }
        //            else {
        //                json_info["encouragement_cdf"] =
        //                this->encouragement_cdf;
        //            }
        //            json.push_back(json_info);
    }

    string ASISTStudy3InterventionEstimator::get_name() const { return NAME; }

    void ASISTStudy3InterventionEstimator::prepare() {
        this->last_time_step = -1;
        this->containers_initialized = false;
    }

    void ASISTStudy3InterventionEstimator::set_logger(
        const OnlineLoggerPtr& logger) {
        Estimator::set_logger(logger);
        if (const auto& tmp =
                dynamic_pointer_cast<ASISTStudy3InterventionLogger>(logger)) {
            // We store a reference to the logger into a local variable to
            // avoid casting throughout the code.
            this->custom_logger = tmp;
        }
        else {
            throw TomcatModelException(
                "The ASISTStudy3InterventionEstimator requires a "
                "logger of type ASISTStudy3InterventionLogger.");
        }
    }

    double ASISTStudy3InterventionEstimator::get_encouragement_cdf() const {
        auto& encouragement_node =
            dynamic_pointer_cast<ASISTStudy3InterventionModel>(this->model)
                ->get_encouragement_node();
        return encouragement_node->get_cdfs(1, 0, false)(0);
    }

    int ASISTStudy3InterventionEstimator::get_num_encouragements() const {
        auto& encouragement_node =
            dynamic_pointer_cast<ASISTStudy3InterventionModel>(this->model)
                ->get_encouragement_node();
        return (int)encouragement_node->get_assignment()(0, 0);
    }

    void ASISTStudy3InterventionEstimator::clear_active_unspoken_marker(
        int player_order) {

        if (this->active_markers[player_order] ==
            this->watched_markers[player_order]) {
            // Remove if from the last placed marker as well because we
            // don't want to keep track of it anymore.
            this->watched_markers[player_order] = Marker();
        }

        this->active_markers[player_order] = Marker();
    }

    void ASISTStudy3InterventionEstimator::
        clear_active_no_ask_for_help_critical_victim(int player_order) {
        this->active_no_critical_victim_help_requests[player_order] = false;
    }

    void ASISTStudy3InterventionEstimator::clear_active_no_ask_for_help_threat(
        int player_order) {
        this->active_no_threat_help_requests[player_order] = false;
        this->help_request_room_escape_state[player_order] =
            InterventionState::WATCHED;
        this->help_request_room_escape_timer[player_order] =
            ASK_FOR_HELP_LATENCY;
    }

    void ASISTStudy3InterventionEstimator::clear_active_no_help_on_the_way(
        int player_order) {
        this->active_no_help_on_the_way[player_order] = false;
    }

    //----------------------------------------------------------------------
    // Getters & Setters
    //----------------------------------------------------------------------

    int ASISTStudy3InterventionEstimator::get_last_time_step() const {
        return last_time_step;
    }

    const vector<Marker>&
    ASISTStudy3InterventionEstimator::get_active_unspoken_markers() const {
        return active_markers;
    }

    const vector<bool>& ASISTStudy3InterventionEstimator::
        get_active_no_critical_victim_help_request() const {
        return this->active_no_critical_victim_help_requests;
    }

    vector<bool>
    ASISTStudy3InterventionEstimator::get_active_no_threat_help_request()
        const {
        vector<bool> active;

        for (const auto& state : this->help_request_room_escape_state) {
            active.push_back(state == InterventionState::ACTIVE);
        }

        return active;
    }

    const vector<bool>&
    ASISTStudy3InterventionEstimator::get_active_no_help_on_the_way() const {
        return this->active_no_help_on_the_way;
    }

    //----------------------------------------------------------------------
    // Getters & Setters
    //----------------------------------------------------------------------

} // namespace tomcat::model
