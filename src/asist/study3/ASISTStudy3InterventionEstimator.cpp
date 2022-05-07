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
        const Marker& marker,
        int time_step,
        const EvidenceSet& new_data) {

        const auto& json_dialog =
            new_data.get_dict_like_data()[0][time_step][Labels::DIALOG]
                                         [player_order];

        return EXISTS(
            marker.type,
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

    shared_ptr<Marker> ASISTStudy3InterventionEstimator::get_last_placed_marker(
        int player_order, int time_step, const EvidenceSet& new_data) {

        const auto& json_marker =
            new_data
                .get_dict_like_data()[0][time_step][Labels::LAST_PLACED_MARKERS]
                                     [player_order];

        if (!json_marker.empty()) {
            return make_shared<Marker>(Marker(json_marker));
        }

        return nullptr;
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

    bool ASISTStudy3InterventionEstimator::did_any_player_remove_marker(
        const ASISTStudy3MessageConverter::Marker& marker,
        int time_step,
        const EvidenceSet& new_data) {

        // I changed the logic to consider removals from any player because
        // if any player removes a marker before the intervention is triggered,
        // it means the player who removed the marker is in the same location
        // as the player who placed the marker, and, therefore, the player who
        // placed the marker is very likely to be aware of the removal.
        // I found a concrete case of this situation in a trial.

        for (int player_order = 0; player_order < 3; player_order++) {
            const auto& removed_markers =
                get_removed_markers(player_order, time_step, new_data);

            bool removed = any_of(removed_markers.begin(),
                                  removed_markers.end(),
                                  [&m = marker](const Marker& removed_marker) {
                                      return removed_marker == m;
                                  });

            if (removed) {
                return true;
            }
        }

        return false;
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

    bool ASISTStudy3InterventionEstimator::is_there_another_player_in_same_area(
        int player_order, int time_step, const EvidenceSet& new_data) {

        const string& player_location =
            new_data.get_dict_like_data()[0][time_step][Labels::LOCATIONS]
                                         [player_order]["id"];
        bool around = false;
        for (int i = 0; i < 3; i++) {
            if (i != player_order) {
                const string& other_player_location =
                    new_data.get_dict_like_data()[0][time_step]
                                                 [Labels::LOCATIONS][i]["id"];

                around = player_location == other_player_location;
                if (around) {
                    break;
                }
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

        string threat_id =
            get_active_threat_id(player_order, time_step, new_data);

        const string& player_location =
            new_data.get_dict_like_data()[0][time_step][Labels::LOCATIONS]
                                         [player_order]["id"];

        // The player might be seeing the obstacle but be in a different room.
        // Therefore, not trapped in the room associated with this active
        // threat.
        return !threat_id.empty() && (threat_id == player_location);
    }

    string ASISTStudy3InterventionEstimator::get_active_threat_id(
        int player_order, int time_step, const EvidenceSet& new_data) {

        string threat_id =
            new_data.get_dict_like_data()[0][time_step][Labels::FOV]
                                         [player_order]["collapsed_rubble_id"];
        return threat_id;
    }

    bool ASISTStudy3InterventionEstimator::is_player_inside_room(
        int player_order, int time_step, const EvidenceSet& new_data) {

        return new_data.get_dict_like_data()[0][time_step][Labels::LOCATIONS]
                                            [player_order]["room"];
    }

    bool ASISTStudy3InterventionEstimator::is_engineer_in_same_room(
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

    bool ASISTStudy3InterventionEstimator::should_watch_marker_type(
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
        int time_step, const EvidenceSet& new_data, const string& threat_id) {

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
    }

    void
    ASISTStudy3InterventionEstimator::estimate(const EvidenceSet& new_data) {

        this->initialize_containers(new_data);

        this->first_mission = new_data.get_metadata()[0]["mission_order"] == 1;

        for (int t = 0; t < new_data.get_time_steps(); t++) {
            this->estimate_motivation_intervention(t, new_data);

            for (int player_order = 0; player_order < 3; player_order++) {
                this->update_communication(player_order, t, new_data);
                this->estimate_marker_intervention(player_order, t, new_data);
                this->estimate_help_request_intervention(
                    player_order, t, new_data);
                this->estimate_help_request_reply_intervention(
                    player_order, t, new_data);
            }
        }

        this->last_time_step += new_data.get_time_steps();
    }

    void ASISTStudy3InterventionEstimator::initialize_containers(
        const EvidenceSet& new_data) {
        if (!this->containers_initialized) {
            this->marker_intervention_state =
                vector<InterventionState>(3, InterventionState::NONE);
            this->help_request_critical_victim_intervention_state =
                vector<InterventionState>(3, InterventionState::NONE);
            this->help_request_critical_victim_intervention_timer =
                vector<int>(3, 0);
            this->help_request_room_escape_intervention_state =
                vector<InterventionState>(3, InterventionState::NONE);
            this->help_request_room_escape_intervention_timer =
                vector<int>(3, 0);
            this->latest_active_threat_id = vector<string>(3, "");
            this->help_request_reply_intervention_state =
                vector<InterventionState>(3, InterventionState::NONE);
            this->help_request_reply_intervention_timer = vector<int>(3, 0);

            this->watched_marker = vector<shared_ptr<Marker>>(3, nullptr);
            this->active_marker = vector<shared_ptr<Marker>>(3, nullptr);
            this->mentioned_marker_types = vector<unordered_set<MarkerType>>(3);
            this->recently_mentioned_critical_victim = vector<bool>(3, false);
            this->recently_mentioned_help_request = vector<bool>(3, false);

            this->containers_initialized = true;
        }
    }

    void ASISTStudy3InterventionEstimator::update_communication(
        int player_order, int time_step, const EvidenceSet& new_data) {

        if (did_player_change_area(player_order, time_step, new_data)) {
            // Recent mention to tracked information is cleared when the
            // player changes location.
            this->mentioned_marker_types[player_order].clear();
            this->recently_mentioned_critical_victim[player_order] = false;
            this->recently_mentioned_help_request[player_order] = false;
        }

        const auto& marker_types =
            get_mentioned_marker_types(player_order, time_step, new_data);

        this->mentioned_marker_types[player_order].insert(marker_types.begin(),
                                                          marker_types.end());

        const auto& json_dialog =
            new_data.get_dict_like_data()[0][time_step][Labels::DIALOG]
                                         [player_order];

        if ((bool)json_dialog["critical_victim"]) {
            this->recently_mentioned_critical_victim[player_order] = true;
        }
        if ((bool)json_dialog["help_needed"]) {
            this->recently_mentioned_help_request[player_order] = true;
        }
    }

    void ASISTStudy3InterventionEstimator::estimate_motivation_intervention(
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

    void ASISTStudy3InterventionEstimator::estimate_marker_intervention(
        int player_order, int time_step, const EvidenceSet& new_data) {

        if (this->marker_intervention_state[player_order] ==
            InterventionState::ACTIVE) {
            // Intervention was not delivered.
            this->restart_marker_intervention(player_order);
        }

        if (this->marker_intervention_state[player_order] ==
            InterventionState::NONE) {

            auto new_marker =
                get_last_placed_marker(player_order, time_step, new_data);
            if (new_marker && should_watch_marker_type(new_marker->type)) {
                bool marker_mentioned_recently =
                    EXISTS(new_marker->type,
                           this->mentioned_marker_types[player_order]);
                if (marker_mentioned_recently) {
                    this->custom_logger->log_hinder_marker_intervention(
                        this->last_time_step + time_step + 1,
                        player_order,
                        *new_marker);
                }
                else {
                    this->marker_intervention_state[player_order] =
                        InterventionState::WATCHED;
                    this->watched_marker[player_order] = new_marker;

                    this->custom_logger->log_watch_marker_intervention(
                        this->last_time_step + time_step + 1,
                        player_order,
                        *new_marker);
                }
            }
        }
        else {
            // Check if it can be canceled
            bool marker_removed = did_any_player_remove_marker(
                *this->watched_marker[player_order], time_step, new_data);

            bool marker_mentioned = did_player_speak_about_marker(
                player_order,
                *this->watched_marker[player_order],
                time_step,
                new_data);

            if (marker_removed || marker_mentioned) {
                this->marker_intervention_state[player_order] =
                    InterventionState::NONE;

                this->custom_logger->log_cancel_marker_intervention(
                    this->last_time_step + time_step + 1,
                    player_order,
                    *this->watched_marker[player_order],
                    marker_removed,
                    marker_mentioned);
            }

            // Check for activation
            if (this->marker_intervention_state[player_order] ==
                InterventionState::WATCHED) {

                auto new_marker =
                    get_last_placed_marker(player_order, time_step, new_data);
                if (new_marker) {
                    // Check if the new marker is far enough from the previous
                    // placed marker.
                    if (new_marker->distance_to(
                            *this->watched_marker[player_order]) <=
                        VICINITY_MAX_RADIUS) {
                        if (should_watch_marker_type(new_marker->type)) {
                            this->watched_marker[player_order] = new_marker;
                        }

                        this->custom_logger
                            ->log_marker_too_close_marker_intervention(
                                this->last_time_step + time_step + 1,
                                player_order,
                                *new_marker,
                                *this->watched_marker[player_order]);

                        new_marker = nullptr;
                    }
                }

                bool area_changed =
                    did_player_change_area(player_order, time_step, new_data);
                bool victim_interaction = did_player_interact_with_victim(
                    player_order, time_step, new_data);

                if (area_changed || victim_interaction || new_marker) {
                    this->marker_intervention_state[player_order] =
                        InterventionState::ACTIVE;

                    this->active_marker[player_order] =
                        move(this->watched_marker[player_order]);

                    this->custom_logger->log_activate_marker_intervention(
                        this->last_time_step + time_step + 1,
                        player_order,
                        *this->active_marker[player_order],
                        area_changed,
                        victim_interaction,
                        new_marker != nullptr);

                    // If a new marker triggered the intervention activation, it
                    // must be added to the watch list.
                    if (new_marker &&
                        should_watch_marker_type(new_marker->type)) {
                        this->watched_marker[player_order] = new_marker;

                        this->custom_logger->log_watch_marker_intervention(
                            this->last_time_step + time_step + 1,
                            player_order,
                            *new_marker);
                    }
                }
            }
        }
    }

    void ASISTStudy3InterventionEstimator::estimate_help_request_intervention(
        int player_order, int time_step, const EvidenceSet& new_data) {
        this->estimate_help_request_critical_victim_intervention(
            player_order, time_step, new_data);
        this->estimate_help_request_room_escape_intervention(
            player_order, time_step, new_data);
    }

    void ASISTStudy3InterventionEstimator::
        estimate_help_request_critical_victim_intervention(
            int player_order, int time_step, const EvidenceSet& new_data) {

        if (this->help_request_critical_victim_intervention_state
                [player_order] == InterventionState::NONE) {
            bool help_needed = does_player_need_help_to_wake_victim(
                player_order, time_step, new_data);
            bool help_requested =
                did_player_ask_for_help(player_order, time_step, new_data);
            bool mentioned_critical_victim =
                did_player_speak_about_critical_victim(
                    player_order, time_step, new_data);
            bool is_alone = !is_there_another_player_in_same_area(
                player_order, time_step, new_data);

            if (help_needed && !help_requested && !mentioned_critical_victim &&
                is_alone) {
                bool recent_mention_to_help =
                    this->recently_mentioned_help_request[player_order];
                bool recent_mention_to_critical_victim =
                    this->recently_mentioned_critical_victim[player_order];

                if (recent_mention_to_help ||
                    recent_mention_to_critical_victim) {
                    this->custom_logger
                        ->log_hinder_help_request_critical_victim_intervention(
                            this->last_time_step + time_step + 1, player_order);
                }
                else {
                    this->help_request_critical_victim_intervention_state
                        [player_order] = InterventionState::WATCHED;
                    this->help_request_critical_victim_intervention_timer
                        [player_order] = HELP_REQUEST_LATENCY;

                    this->custom_logger
                        ->log_watch_help_request_critical_victim_intervention(
                            this->last_time_step + time_step + 1, player_order);
                }
            }
        }
        else {
            // Check if it can be canceled
            bool help_requested =
                did_player_ask_for_help(player_order, time_step, new_data);
            bool mentioned_critical_victim =
                did_player_speak_about_critical_victim(
                    player_order, time_step, new_data);
            bool changed_area =
                did_player_change_area(player_order, time_step, new_data);
            int is_alone = !is_there_another_player_in_same_area(
                player_order, time_step, new_data);

            if (help_requested || mentioned_critical_victim || changed_area ||
                !is_alone) {
                this->help_request_critical_victim_intervention_state
                    [player_order] = InterventionState::NONE;

                this->custom_logger
                    ->log_cancel_help_request_critical_victim_intervention(
                        this->last_time_step + time_step + 1,
                        player_order,
                        changed_area,
                        help_requested,
                        mentioned_critical_victim,
                        !is_alone);
            }

            if (this->help_request_critical_victim_intervention_state
                    [player_order] == InterventionState::WATCHED) {
                // Decrement counter
                this->help_request_critical_victim_intervention_timer
                    [player_order] -= 1;
                if (this->help_request_critical_victim_intervention_timer
                        [player_order] == 0) {
                    this->help_request_critical_victim_intervention_state
                        [player_order] = InterventionState::ACTIVE;

                    this->custom_logger
                        ->log_activate_help_request_critical_victim_intervention(
                            this->last_time_step + time_step + 1,
                            player_order,
                            HELP_REQUEST_LATENCY);
                }
            }
        }
    }

    void ASISTStudy3InterventionEstimator::
        estimate_help_request_room_escape_intervention(
            int player_order, int time_step, const EvidenceSet& new_data) {

        if (this->help_request_room_escape_intervention_state[player_order] ==
            InterventionState::NONE) {
            bool help_requested =
                did_player_ask_for_help(player_order, time_step, new_data);
            bool is_trapped = does_player_need_help_to_exit_room(
                player_order, time_step, new_data);
            bool is_engineer_in_room =
                is_engineer_in_same_room(player_order, time_step, new_data);

            if (!help_requested && is_trapped && !is_engineer_in_room) {
                string threat_id =
                    get_active_threat_id(player_order, time_step, new_data);
                bool is_being_released =
                    is_player_being_released(time_step, new_data, threat_id);
                bool recent_mention_to_help =
                    this->recently_mentioned_help_request[player_order];

                if (recent_mention_to_help || is_being_released) {
                    this->custom_logger
                        ->log_hinder_help_request_room_escape_intervention(
                            this->last_time_step + time_step + 1,
                            player_order,
                            recent_mention_to_help,
                            is_being_released);
                }
                else {
                    this->help_request_room_escape_intervention_state
                        [player_order] = InterventionState::WATCHED;
                    this->help_request_room_escape_intervention_timer
                        [player_order] = HELP_REQUEST_LATENCY;
                    this->latest_active_threat_id[player_order] =
                        get_active_threat_id(player_order, time_step, new_data);

                    this->custom_logger
                        ->log_watch_help_request_room_escape_intervention(
                            this->last_time_step + time_step + 1, player_order);
                }
            }
        }
        else {
            // Check if it can be canceled
            bool help_requested =
                did_player_ask_for_help(player_order, time_step, new_data);
            bool left_room =
                !is_player_inside_room(player_order, time_step, new_data);
            bool is_engineer_in_room =
                is_engineer_in_same_room(player_order, time_step, new_data);
            bool being_released = this->is_player_being_released(
                time_step,
                new_data,
                this->latest_active_threat_id[player_order]);

            if (help_requested || left_room || is_engineer_in_room ||
                being_released) {
                this->help_request_room_escape_intervention_state
                    [player_order] = InterventionState::NONE;

                this->custom_logger
                    ->log_cancel_help_request_room_escape_intervention(
                        this->last_time_step + time_step + 1,
                        player_order,
                        left_room,
                        help_requested,
                        being_released,
                        is_engineer_in_room);
            }

            if (this->help_request_room_escape_intervention_state
                    [player_order] == InterventionState::WATCHED) {
                // Decrement counter
                this->help_request_room_escape_intervention_timer
                    [player_order] -= 1;
                if (this->help_request_room_escape_intervention_timer
                        [player_order] == 0) {
                    this->help_request_room_escape_intervention_state
                        [player_order] = InterventionState::ACTIVE;

                    this->custom_logger
                        ->log_activate_help_request_room_escape_intervention(
                            this->last_time_step + time_step + 1,
                            player_order,
                            HELP_REQUEST_LATENCY);
                }
            }
        }
    }

    void
    ASISTStudy3InterventionEstimator::estimate_help_request_reply_intervention(
        int player_order, int time_step, const EvidenceSet& new_data) {

        if (this->help_request_reply_intervention_state[player_order] ==
            InterventionState::NONE) {
            bool help_requested =
                did_player_ask_for_help(player_order, time_step, new_data);

            if (help_requested) {
                this->help_request_reply_intervention_state[player_order] =
                    InterventionState::WATCHED;
                this->help_request_reply_intervention_timer[player_order] =
                    HELP_REQUEST_REPLY_LATENCY;

                this->custom_logger->log_watch_help_request_reply_intervention(
                    this->last_time_step + time_step + 1, player_order);
            }
        }
        else {
            // Check if it can be canceled
            bool area_changed =
                did_player_change_area(player_order, time_step, new_data);
            int helper_player_order =
                get_helper_player_order(player_order, time_step, new_data);
            bool help_request_answered = helper_player_order >= 0;

            if (help_request_answered || area_changed) {
                this->help_request_reply_intervention_state[player_order] =
                    InterventionState::NONE;

                this->custom_logger->log_cancel_help_request_reply_intervention(
                    this->last_time_step + time_step + 1,
                    player_order,
                    helper_player_order,
                    area_changed,
                    help_request_answered);

                if (area_changed) {
                    this->help_request_reply_intervention_state[player_order] =
                        InterventionState::NONE;
                }
                else {
                    // Restart the counter. Even if other players answered to
                    // the help request, they might forget about it.
                    this->help_request_reply_intervention_state[player_order] =
                        InterventionState::WATCHED;
                    this->help_request_reply_intervention_timer[player_order] =
                        HELP_REQUEST_REPLY_LATENCY;

                    this->custom_logger
                        ->log_watch_help_request_reply_intervention(
                            this->last_time_step + time_step + 1, player_order);
                }
            }

            if (this->help_request_reply_intervention_state[player_order] ==
                InterventionState::WATCHED) {
                // Decrement counter
                this->help_request_reply_intervention_timer[player_order] -= 1;
                if (this->help_request_reply_intervention_timer[player_order] ==
                    0) {
                    this->help_request_reply_intervention_state[player_order] =
                        InterventionState::ACTIVE;

                    this->custom_logger
                        ->log_activate_help_request_reply_intervention(
                            this->last_time_step + time_step + 1,
                            player_order,
                            HELP_REQUEST_REPLY_LATENCY);
                }
            }
        }
    }

    void
    ASISTStudy3InterventionEstimator::get_info(nlohmann::json& json) const {}

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

    const Marker& ASISTStudy3InterventionEstimator::get_active_marker(
        int player_order) const {
        return *this->active_marker.at(player_order);
    }

    bool ASISTStudy3InterventionEstimator::is_marker_intervention_active(
        int player_order) {
        return this->marker_intervention_state[player_order] ==
               InterventionState::ACTIVE;
    }

    bool ASISTStudy3InterventionEstimator::
        is_help_request_critical_victim_intervention_active(int player_order) {
        return this->help_request_critical_victim_intervention_state
                   [player_order] == InterventionState::ACTIVE;
    }

    bool ASISTStudy3InterventionEstimator::
        is_help_request_room_escape_intervention_active(int player_order) {
        return this->help_request_room_escape_intervention_state
                   [player_order] == InterventionState::ACTIVE;
    }

    bool
    ASISTStudy3InterventionEstimator::is_help_request_reply_intervention_active(
        int player_order) {
        return this->help_request_reply_intervention_state[player_order] ==
               InterventionState::ACTIVE;
    }

    void ASISTStudy3InterventionEstimator::restart_marker_intervention(
        int player_order) {

        // This is a single-shot intervention. Once delivered to a participant,
        // it is no longer tracked unless there's a different marker in the
        // watching list.
        if (this->watched_marker[player_order]) {
            this->marker_intervention_state[player_order] =
                InterventionState::WATCHED;
        }
        else {
            this->marker_intervention_state[player_order] =
                InterventionState::NONE;
        }
    }

    void ASISTStudy3InterventionEstimator::
        restart_help_request_critical_victim_intervention(int player_order) {
        // This is a recurrent intervention. Once delivered to a participant, we
        // keep watching for it until it gets canceled.
        this->help_request_critical_victim_intervention_state[player_order] =
            InterventionState::WATCHED;
        this->help_request_critical_victim_intervention_timer[player_order] =
            HELP_REQUEST_LATENCY;
    }

    void ASISTStudy3InterventionEstimator::
        restart_help_request_room_escape_intervention(int player_order) {
        // This is a recurrent intervention. Once delivered to a participant, we
        // keep watching for it until it gets canceled.
        this->help_request_room_escape_intervention_state[player_order] =
            InterventionState::WATCHED;
        this->help_request_room_escape_intervention_timer[player_order] =
            HELP_REQUEST_LATENCY;
    }

    void
    ASISTStudy3InterventionEstimator::restart_help_request_reply_intervention(
        int player_order) {
        // This is a recurrent intervention. Once delivered to a participant, we
        // keep watching for it until it gets canceled.
        this->help_request_reply_intervention_state[player_order] =
            InterventionState::WATCHED;
        this->help_request_reply_intervention_timer[player_order] =
            HELP_REQUEST_REPLY_LATENCY;
    }

    //----------------------------------------------------------------------
    // Getters & Setters
    //----------------------------------------------------------------------

    int ASISTStudy3InterventionEstimator::get_last_time_step() const {
        return last_time_step;
    }

    //----------------------------------------------------------------------
    // Getters & Setters
    //----------------------------------------------------------------------

} // namespace tomcat::model
