#include "ASISTStudy3InterventionEstimator.h"

#include <algorithm>

#include <fmt/format.h>

#include "asist/study3/ASISTStudy3InterventionModel.h"
#include "asist/study3/ASISTStudy3MessageConverter.h"
#include "utils/EigenExtensions.h"
#include "utils/JSONChecker.h"

namespace tomcat {
    namespace model {

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

            if (unspoken_marker.type == MarkerType::NO_VICTIM &&
                (bool)json_dialog["no_victim"]) {
                return true;
            }
            else if (unspoken_marker.type == MarkerType::REGULAR_VICTIM &&
                     (bool)json_dialog["regular_victim"]) {
                return true;
            }
            else if (unspoken_marker.type == MarkerType::VICTIM_C &&
                     (bool)json_dialog["critical_victim"]) {
                return true;
            }
            else if (unspoken_marker.type == MarkerType::VICTIM_A &&
                     (bool)json_dialog["victim_a"]) {
                return true;
            }
            else if (unspoken_marker.type == MarkerType::VICTIM_B &&
                     (bool)json_dialog["victim_b"]) {
                return true;
            }
            else if (unspoken_marker.type == MarkerType::RUBBLE &&
                     (bool)json_dialog["obstacle"]) {
                return true;
            }
            else if (unspoken_marker.type == MarkerType::THREAT_ROOM &&
                     (bool)json_dialog["threat"]) {
                return true;
            }
            else if (unspoken_marker.type == MarkerType::SOS &&
                     (bool)json_dialog["help_needed"]) {
                return true;
            }

            return false;
        }

        Marker ASISTStudy3InterventionEstimator::get_last_placed_marker(
            int player_order, int time_step, const EvidenceSet& new_data) {

            const auto& json_marker =
                new_data.get_dict_like_data()[0][time_step]
                                             [Labels::LAST_PLACED_MARKERS]
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
                new_data.get_dict_like_data()[0][time_step]
                                             [Labels::PLAYER_POSITIONS]
                                             [player_order]);
            return player_pos.distance_to(position) > max_distance;
        }

        vector<Marker> ASISTStudy3InterventionEstimator::get_removed_markers(
            int player_order, int time_step, const EvidenceSet& new_data) {

            vector<Marker> removed_markers;
            const auto& json_markers =
                new_data
                    .get_dict_like_data()[0][time_step][Labels::REMOVED_MARKERS]
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

            for (const auto& removed_marker :
                 get_removed_markers(player_order, time_step, new_data)) {

                if (marker == removed_marker) {
                    return true;
                }
            }

            return false;
        }

        bool
        ASISTStudy3InterventionEstimator::does_player_need_help_to_wake_victim(
            int player_order, int time_step, const EvidenceSet& new_data) {

            double distance =
                new_data.get_dict_like_data()[0][time_step][Labels::FOV]
                                             [player_order]
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
                        new_data.get_dict_like_data()[0][time_step]
                                                     [Labels::LOCATIONS]
                                                     [player_order]["id"];

                    around = player_location == other_player_location;
                }
            }

            return around;
        }

        bool ASISTStudy3InterventionEstimator::
            did_player_speak_about_critical_victim(
                int player_order, int time_step, const EvidenceSet& new_data) {

            const auto& json_dialog =
                new_data.get_dict_like_data()[0][time_step][Labels::DIALOG]
                                             [player_order];

            return json_dialog["critical_victim"];
        }

        bool
        ASISTStudy3InterventionEstimator::does_player_need_help_to_exit_room(
            int player_order, int time_step, const EvidenceSet& new_data) {

            return !get_threat_id(player_order, time_step, new_data).empty();
        }

        string ASISTStudy3InterventionEstimator::get_threat_id(
            int player_order, int time_step, const EvidenceSet& new_data) {

            string threat_id = new_data
                .get_dict_like_data()[0][time_step][Labels::FOV][player_order]
                                     ["collapsed_rubble_id"];
            return threat_id;
        }

        bool ASISTStudy3InterventionEstimator::is_player_being_released(
            int player_order, int time_step, const EvidenceSet& new_data) {

            string threat_id = get_threat_id(player_order, time_step, new_data);

            const string& threat_id_being_removed =
                new_data.get_dict_like_data()
                    [0][time_step][Labels::RUBBLE_COLLAPSE]
                    ["destruction_interaction_collapsed_rubble_id"];

            if (threat_id.empty() || threat_id_being_removed.empty()) {
                return false;
            }

            return threat_id == threat_id_being_removed;
        }

        bool ASISTStudy3InterventionEstimator::is_player_in_room(
            int player_order, int time_step, const EvidenceSet& new_data) {

            return new_data
                .get_dict_like_data()[0][time_step][Labels::LOCATIONS]
                                     [player_order]["room"];
        }

        bool ASISTStudy3InterventionEstimator::is_engineer_around(
            int player_order, int time_step, const EvidenceSet& new_data) {

            int engineer_order = new_data.get_metadata()[0]["engineer_order"];

            bool around = false;
            if (player_order == engineer_order) {
                around = true;
            }
            else {
                // Check if the engineer is in the same location as the player
                const string& player_location =
                    new_data
                        .get_dict_like_data()[0][time_step][Labels::LOCATIONS]
                                             [player_order]["id"];

                const string& engineer_location =
                    new_data
                        .get_dict_like_data()[0][time_step][Labels::LOCATIONS]
                                             [engineer_order]["id"];

                around = player_location == engineer_location;
            }

            return around;
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

        void ASISTStudy3InterventionEstimator::estimate(
            const EvidenceSet& new_data) {

            this->initialize_containers(new_data);

            this->first_mission =
                new_data.get_metadata()[0]["mission_order"] == 1;
            this->estimate_motivation(new_data);
            this->estimate_communication_marker(new_data);
            this->estimate_help_request(new_data);
            this->last_time_step += new_data.get_time_steps();
        }

        void ASISTStudy3InterventionEstimator::initialize_containers(
            const EvidenceSet& new_data) {
            if (!this->containers_initialized) {
                this->watched_markers = vector<Marker>(3);
                this->watched_critical_victims = vector<int>(3, -1);
                this->watched_threats = vector<pair<string, int>>(3, {"", -1});
                this->active_markers = vector<Marker>(3);
                this->active_no_critical_victim_help_requests =
                    vector<bool>(3, false);
                this->active_no_threat_help_requests = vector<bool>(3, false);
                this->containers_initialized = true;
            }
        }

        void ASISTStudy3InterventionEstimator::estimate_motivation(
            const EvidenceSet& new_data) {
            const auto& metadata = new_data.get_metadata();

            check_field(metadata[0], "mission_order");

            auto& encouragement_node =
                dynamic_pointer_cast<ASISTStudy3InterventionModel>(this->model)
                    ->get_encouragement_node();

            int increments = 0;
            if (metadata[0]["mission_order"] == 1) {
                if (this->last_time_step < 0) {
                    encouragement_node->set_assignment(
                        Eigen::VectorXd::Zero(1));
                    this->custom_logger->log_watch_motivation_intervention(0);
                }

                for (int t = 0; t < new_data.get_time_steps(); t++) {
                    increments +=
                        (int)new_data
                            .get_dict_like_data()[0][t][Labels::ENCOURAGEMENT];
                }

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
                    encouragement_node->set_assignment(
                        Eigen::VectorXd::Zero(1));
                }
            }
        }

        void ASISTStudy3InterventionEstimator::estimate_communication_marker(
            const EvidenceSet& new_data) {

            for (int t = 0; t < new_data.get_time_steps(); t++) {
                for (int player_order = 0; player_order < 3; player_order++) {

                    if (!this->watched_markers[player_order].is_none()) {
                        // Cancel watched marker if possible
                        bool marker_removal = did_player_remove_marker(
                            this->watched_markers[player_order],
                            player_order,
                            t,
                            new_data);
                        bool speech = did_player_speak_about_marker(
                            player_order,
                            this->watched_markers[player_order],
                            t,
                            new_data);

                        if (marker_removal || speech) {
                            this->custom_logger
                                ->log_cancel_communication_marker_intervention(
                                    this->last_time_step + t + 1,
                                    player_order,
                                    this->watched_markers[player_order],
                                    speech,
                                    marker_removal);
                            this->watched_markers[player_order] = Marker();
                        }
                    }

                    Marker new_marker =
                        get_last_placed_marker(player_order, t, new_data);
                    bool area_changed =
                        did_player_change_area(player_order, t, new_data);
                    bool victim_interaction = did_player_interact_with_victim(
                        player_order, t, new_data);
                    bool marker_placed = !new_marker.is_none();
                    bool count_as_new_marker = false;
                    if (!this->watched_markers[player_order].is_none() &&
                        marker_placed) {
                        count_as_new_marker =
                            new_marker.position.distance_to(
                                this->watched_markers[player_order].position) >
                            VICINITY_MAX_RADIUS;
                    }

                    bool active_intervention =
                        !this->watched_markers[player_order].is_none() &&
                        (area_changed || victim_interaction ||
                         (marker_placed && count_as_new_marker));

                    if (active_intervention) {
                        this->active_markers[player_order] =
                            this->watched_markers[player_order];

                        this->custom_logger
                            ->log_activate_communication_marker_intervention(
                                this->last_time_step + t + 1,
                                player_order,
                                this->watched_markers[player_order],
                                area_changed,
                                victim_interaction,
                                marker_placed);

                        this->watched_markers[player_order] = Marker();
                    }

                    if (marker_placed) {
                        // Watch the new marker for intervention
                        this->watched_markers[player_order] = new_marker;

                        this->custom_logger
                            ->log_watch_communication_marker_intervention(
                                this->last_time_step + t + 1,
                                player_order,
                                new_marker);
                    }
                }
            }
        }

        void ASISTStudy3InterventionEstimator::estimate_help_request(
            const EvidenceSet& new_data) {
            this->estimate_critical_victim_help_request(new_data);
            this->estimate_threat_help_request(new_data);
        }

        void
        ASISTStudy3InterventionEstimator::estimate_critical_victim_help_request(
            const EvidenceSet& new_data) {
            for (int t = 0; t < new_data.get_time_steps(); t++) {
                for (int player_order = 0; player_order < 3; player_order++) {

                    if (this->watched_critical_victims[player_order] >= 0) {
                        // Cancel intervention if possible
                        bool area_changed =
                            did_player_change_area(player_order, t, new_data);
                        bool help_requested =
                            did_player_ask_for_help(player_order, t, new_data);
                        bool mention_to_critical_victim =
                            did_player_speak_about_critical_victim(
                                player_order, t, new_data);
                        int other_players_around =
                            is_there_another_player_around(
                                player_order, t, new_data);
                        if (area_changed || help_requested ||
                            mention_to_critical_victim ||
                            other_players_around) {
                            this->active_no_critical_victim_help_requests
                                [player_order] = false;
                            this->watched_critical_victims[player_order] = -1;

                            this->custom_logger
                                ->log_cancel_ask_for_help_critical_victim_intervention(
                                    this->last_time_step + t + 1,
                                    player_order,
                                    area_changed,
                                    help_requested,
                                    mention_to_critical_victim,
                                    other_players_around);
                        }
                    }

                    // Watch and activate intervention
                    if (does_player_need_help_to_wake_victim(
                            player_order, t, new_data) &&
                        !is_there_another_player_around(
                            player_order, t, new_data)) {
                        if (this->watched_critical_victims[player_order] >= 0) {
                            // We are already watching this player for help
                            // request regarding critical victim awakening
                            if (this->last_time_step + t + 1 -
                                    this->watched_critical_victims
                                        [player_order] >
                                ASK_FOR_HELP_LATENCY) {
                                // It has passed enough time. Restart watching
                                // time and activate intervention.
                                this->watched_critical_victims[player_order] =
                                    this->last_time_step + t + 1;
                                this->active_no_critical_victim_help_requests
                                    [player_order] = true;

                                this->custom_logger
                                    ->log_activate_ask_for_help_critical_victim_intervention(
                                        this->last_time_step + t + 1,
                                        player_order,
                                        ASK_FOR_HELP_LATENCY);
                            }
                        }
                        else {
                            // This player was not being watched for help
                            // request regarding saving a critical victim
                            this->watched_critical_victims[player_order] =
                                this->last_time_step + t + 1;

                            this->custom_logger
                                ->log_watch_ask_for_help_critical_victim_intervention(
                                    this->last_time_step + t + 1, player_order);
                        }
                    }
                }
            }
        }

        void ASISTStudy3InterventionEstimator::estimate_threat_help_request(
            const EvidenceSet& new_data) {
            for (int t = 0; t < new_data.get_time_steps(); t++) {
                for (int player_order = 0; player_order < 3; player_order++) {

                    bool is_in_room =
                        is_player_in_room(player_order, t, new_data);

                    bool canceled_intervention = false;
                    if (!this->watched_threats[player_order].first.empty()) {
                        // Cancel intervention if possible
                        bool help_requested =
                            did_player_ask_for_help(player_order, t, new_data);
                        bool being_released =
                            is_player_being_released(player_order, t, new_data);
                        if (!is_in_room || help_requested || being_released) {
                            this->active_no_threat_help_requests[player_order] =
                                false;
                            this->watched_threats[player_order] = {"", -1};
                            canceled_intervention = true;

                            this->custom_logger
                                ->log_cancel_ask_for_help_threat_intervention(
                                    this->last_time_step + t + 1,
                                    player_order,
                                    !is_in_room,
                                    help_requested,
                                    being_released);
                        }
                    }

                    // Watch and activate
                    bool is_trapped = does_player_need_help_to_exit_room(
                        player_order, t, new_data);
                    bool is_engineer_nearby =
                        is_engineer_around(player_order, t, new_data);
                    if (!canceled_intervention && is_trapped &&
                        !is_engineer_nearby && is_in_room) {
                        string threat_id =
                            get_threat_id(player_order, t, new_data);
                        if (this->watched_threats[player_order].first.empty()) {
                            // This player was not being watched for help
                            // request regarding exiting a threat room
                            this->watched_threats[player_order] = {
                                threat_id, this->last_time_step + t + 1};

                            this->custom_logger
                                ->log_watch_ask_for_help_threat_intervention(
                                    this->last_time_step + t + 1, player_order);
                        }
                        else {
                            // We are already watching this player for help
                            // request regarding exiting a threat room
                            if (this->last_time_step + t + 1 -
                                    this->watched_threats[player_order].second >
                                ASK_FOR_HELP_LATENCY) {
                                // It has passed enough time. Restart watching
                                // time and activate intervention.
                                this->watched_threats[player_order] = {
                                    threat_id, this->last_time_step + t + 1};
                                this->active_no_threat_help_requests
                                    [player_order] = true;

                                this->custom_logger
                                    ->log_activate_ask_for_help_threat_intervention(
                                        this->last_time_step + t + 1,
                                        player_order,
                                        ASK_FOR_HELP_LATENCY);
                            }
                        }
                    }
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

        string ASISTStudy3InterventionEstimator::get_name() const {
            return NAME;
        }

        void ASISTStudy3InterventionEstimator::prepare() {
            this->last_time_step = -1;
            this->containers_initialized = false;
        }

        void ASISTStudy3InterventionEstimator::set_logger(
            const OnlineLoggerPtr& logger) {
            Estimator::set_logger(logger);
            if (const auto& tmp =
                    dynamic_pointer_cast<ASISTStudy3InterventionLogger>(
                        logger)) {
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
            clear_active_ask_for_help_critical_victim(int player_order) {
            this->active_no_critical_victim_help_requests[player_order] = false;
        }

        void ASISTStudy3InterventionEstimator::clear_active_ask_for_help_threat(
            int player_order) {
            this->active_no_threat_help_requests[player_order] = false;
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

        const vector<bool>&
        ASISTStudy3InterventionEstimator::get_active_no_threat_help_request()
            const {
            return this->active_no_threat_help_requests;
        }

        //        void ASISTStudy3InterventionEstimator::parse_map(
        //            const std::string& map_filepath) {
        //            fstream map_file;
        //            map_file.open(map_filepath);
        //            if (map_file.is_open()) {
        //                nlohmann::json json_map =
        //                nlohmann::json::parse(map_file);
        //
        //                for (const auto& location : json_map["locations"]) {
        //                    const string& type = location["type"];
        //                    const string& id = location["id"];
        //
        //                    if (type == "room") {
        //                        this->room_id_to_idx[id] =
        //                        this->room_ids.size();
        //                        this->room_ids.push_back(id);
        //                    }
        //                }
        //            }
        //            else {
        //                throw TomcatModelException(
        //                    fmt::format("File {} could not be open.",
        //                    map_filepath));
        //            }
        //        }
        //
        //        void
        //        ASISTStudy3InterventionEstimator::create_belief_estimators(
        //            const string& threat_room_model_filepath) {
        //            this->create_threat_room_belief_estimators(
        //                threat_room_model_filepath);
        //        }
        //
        //        void
        //        ASISTStudy3InterventionEstimator::create_threat_room_belief_estimators(
        //            const string& threat_room_model_filepath) {
        //
        //            // Since we are using SumProduct, we can use the same
        //            model for all
        //            // estimators since the SP estimator does not uses the
        //            model
        //            // directly but a factor graph created from the model.
        //            DBNPtr belief_model = make_shared<DynamicBayesNet>(
        //                DynamicBayesNet::create_from_json(threat_room_model_filepath));
        //            belief_model->unroll(3, true);
        //
        //            this->threat_room_belief_estimators =
        //                vector<vector<SumProductEstimator>>(NUM_ROLES);
        //            for (int i = 0; i < NUM_ROLES; i++) {
        //                for (int j = 0; j < this->room_ids.size(); j++) {
        //                    // Instantiate a new belief model.
        //                    SumProductEstimator estimator(belief_model, 0,
        //                    "Belief");
        //                    this->threat_room_belief_estimators[i].push_back(estimator);
        //                }
        //            }
        //        }
        //
        //        void
        //        ASISTStudy3InterventionEstimator::update_threat_room_beliefs(
        //            const EvidenceSet& new_data) {
        //            for (int i = 0; i < NUM_ROLES; i++) {
        //                for (int j = 0; j < this->room_ids.size(); j++) {
        //                    EvidenceSet threat_room_data;
        //                    string original_label =
        //                        fmt::format("ThreatMarkerInRoom#{}",
        //                        this->room_ids[j]);
        //                    threat_room_data.add_data("ThreatMarkerInRoom",
        //                                              new_data[original_label]);
        //                    original_label =
        //                    fmt::format("ThreatMarkerInFoVInRoom#{}",
        //                                                 this->room_ids[j]);
        //                    threat_room_data.add_data("ThreatMarkerInFoV",
        //                                              new_data[original_label]);
        //                    original_label =
        //                    fmt::format("SpeechAboutThreatRoom#{}",
        //                                                 this->room_ids[j]);
        //                    threat_room_data.add_data("SpeechAboutThreat",
        //                                              new_data[original_label]);
        //
        //                    this->threat_room_belief_estimators[i][j].estimate(
        //                        threat_room_data);
        //                }
        //            }
        //        }

        //----------------------------------------------------------------------
        // Getters & Setters
        //----------------------------------------------------------------------

    } // namespace model
} // namespace tomcat
