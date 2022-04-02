#include "ASISTStudy3InterventionEstimator.h"

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
            } else if (unspoken_marker.type == MarkerType::REGULAR_VICTIM &&
                     (bool)json_dialog["regular_victim"]) {
                return true;
            } else if (unspoken_marker.type == MarkerType::VICTIM_C &&
                (bool)json_dialog["critical_victim"]) {
                return true;
            } else if (unspoken_marker.type == MarkerType::VICTIM_A &&
                     (bool)json_dialog["victim_a"]) {
                return true;
            } else if (unspoken_marker.type == MarkerType::VICTIM_B &&
                     (bool)json_dialog["victim_b"]) {
                return true;
            } else if (unspoken_marker.type == MarkerType::RUBBLE &&
                     (bool)json_dialog["obstacle"]) {
                return true;
            } else if (unspoken_marker.type == MarkerType::THREAT_ROOM &&
                     (bool)json_dialog["threat"]) {
                return true;
            } else if (unspoken_marker.type == MarkerType::SOS &&
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
            this->last_time_step += new_data.get_time_steps();

            this->estimate_motivation(new_data);
            this->estimate_unspoken_markers(new_data);
        }

        void ASISTStudy3InterventionEstimator::initialize_containers(
            const EvidenceSet& new_data) {
            if (!this->containers_initialized) {
                this->last_placed_markers = vector<Marker>(3);
                this->active_unspoken_markers = vector<Marker>(3);
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
                for (int t = 0; t < new_data.get_time_steps(); t++) {
                    increments +=
                        (int)new_data
                            .get_dict_like_data()[0][t][Labels::ENCOURAGEMENT];
                }

                encouragement_node->increment_assignment(increments);
            }
            else {
                if (encouragement_node->get_size() == 0) {
                    // The agent crashed and data from mission 1 was lost.
                    // Assume there was no encouragement utterances to force
                    // intervention.
                    encouragement_node->set_assignment(
                        Eigen::VectorXd::Zero(1));
                }
            }
        }

        void ASISTStudy3InterventionEstimator::estimate_unspoken_markers(
            const EvidenceSet& new_data) {

            for (int t = 0; t < new_data.get_time_steps(); t++) {
                for (int player_order = 0; player_order < 3; player_order++) {

                    const auto& unspoken_marker =
                        this->active_unspoken_markers[player_order];
                    if (unspoken_marker.type != MarkerType::NONE) {
                        if (did_player_speak_about_marker(
                                player_order, unspoken_marker, t, new_data)) {
                            this->clear_active_unspoken_marker(player_order);
                        }
                    }

                    // If the player removed the last marker they placed,
                    // remove it from the list.
                    for (const auto& marker :
                         get_removed_markers(player_order, t, new_data)) {

                        if (marker == this->last_placed_markers[player_order]) {
                            // Clear last marker
                            this->last_placed_markers[player_order] = Marker();
                            break;
                        }
                    }

                    // Check if a new marker was placed and needs to be
                    // monitored for communication.
                    Marker new_marker =
                        get_last_placed_marker(player_order, t, new_data);
                    const auto& last_marker =
                        this->last_placed_markers[player_order];
                    if (new_marker.is_none()) {
                        bool cond1 =
                            did_player_change_area(player_order, t, new_data);
                        bool cond2 = did_player_interact_with_victim(
                            player_order, t, new_data);
                        //                        bool cond3 =
                        //                        !last_marker.is_none() &&
                        //                                     is_player_far_apart(player_order,
                        //                                                         last_marker.position,
                        //                                                         MAX_DIST,
                        //                                                         t,
                        //                                                         new_data);
                        if (cond1 || cond2) {
                            this->active_unspoken_markers[player_order] =
                                last_marker;
                        }
                    }
                    else {
                        if (!last_marker.is_none() &&
                            last_marker.type != new_marker.type) {
                            // The player placed a marker that is different
                            // from the marker it had placed before. Add the
                            // previous marker to the list of unspoken
                            // markers.
                            this->active_unspoken_markers[player_order] =
                                last_marker;
                        }

                        this->last_placed_markers[player_order] = new_marker;
                    }
                }
            }
        }

        void
        ASISTStudy3InterventionEstimator::get_info(nlohmann::json& json) const {
            nlohmann::json json_info;
            json_info["name"] = this->get_name();
            if (this->first_mission) {
                json_info["num_encouragements"] = to_string(
                    dynamic_pointer_cast<ASISTStudy3InterventionModel>(
                        this->model)
                        ->get_encouragement_node()
                        ->get_assignment());
            }
            else {
                json_info["encouragement_cdf"] = this->encouragement_cdf;
            }
            json.push_back(json_info);
        }

        string ASISTStudy3InterventionEstimator::get_name() const {
            return NAME;
        }

        void ASISTStudy3InterventionEstimator::prepare() {
            this->last_time_step = -1;
            this->containers_initialized = false;
        }

        double ASISTStudy3InterventionEstimator::get_encouragement_cdf() {
            auto& encouragement_node =
                dynamic_pointer_cast<ASISTStudy3InterventionModel>(this->model)
                    ->get_encouragement_node();
            this->encouragement_cdf =
                encouragement_node->get_cdfs(1, 0, false)(0);
            return this->encouragement_cdf;
        }

        void ASISTStudy3InterventionEstimator::clear_active_unspoken_marker(
            int player_order) {

            if (this->active_unspoken_markers[player_order] ==
                this->last_placed_markers[player_order]) {
                // Remove if from the last placed marker as well because we
                // don't want to keep track of it anymore.
                this->last_placed_markers[player_order] = Marker();
            }

            this->active_unspoken_markers[player_order] = Marker();
        }

        //----------------------------------------------------------------------
        // Getters & Setters
        //----------------------------------------------------------------------

        int ASISTStudy3InterventionEstimator::get_last_time_step() const {
            return last_time_step;
        }

        const vector<Marker>&
        ASISTStudy3InterventionEstimator::get_active_unspoken_markers() const {
            return active_unspoken_markers;
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
