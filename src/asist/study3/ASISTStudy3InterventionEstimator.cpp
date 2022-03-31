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
            int data_point,
            int time_step,
            const EvidenceSet& new_data) {

            const string& marker_type_text =
                new_data
                    .get_dict_like_data()[data_point][time_step]
                                         [Labels::SPOKEN_MARKER][player_order];

            if (!marker_type_text.empty()) {
                MarkerType spoken_marker_type =
                    ASISTStudy3MessageConverter::MARKER_TEXT_TO_TYPE.at(
                        marker_type_text);

                return unspoken_marker.type == spoken_marker_type;
            }

            return false;
        }

        ASISTStudy3InterventionEstimator::Marker
        ASISTStudy3InterventionEstimator::get_last_placed_marker(
            int player_order,
            int data_point,
            int time_step,
            const EvidenceSet& new_data) {

            const auto& json_marker =
                new_data.get_dict_like_data()[data_point][time_step]
                                             [Labels::LAST_PLACED_MARKERS]
                                             [player_order];

            if (!json_marker.empty()) {
                Position pos((double)json_marker["x"],
                             (double)json_marker["z"]);
                MarkerType type =
                    ASISTStudy3MessageConverter::MARKER_TEXT_TO_TYPE.at(
                        (string)json_marker["type"]);

                return Marker(type, pos);
            }

            return Marker();
        }

        bool ASISTStudy3InterventionEstimator::did_player_interact_with_victim(
            int player_order,
            int data_point,
            int time_step,
            const EvidenceSet& new_data) {

            return new_data
                .get_dict_like_data()[data_point][time_step]
                                     [Labels::VICTIM_INTERACTION][player_order];
        }

        bool ASISTStudy3InterventionEstimator::is_player_far_apart(
            int player_order,
            Position position,
            int max_distance,
            int data_point,
            int time_step,
            const EvidenceSet& new_data) {

            double x = new_data.get_dict_like_data()[data_point][time_step]
                                                    [Labels::PLAYER_POSITION]
                                                    [player_order]["x"];
            double z = new_data.get_dict_like_data()[data_point][time_step]
                                                    [Labels::PLAYER_POSITION]
                                                    [player_order]["z"];
            Position player_pos(x, z);

            return player_pos.distance_to(position) > max_distance;
        }

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        void ASISTStudy3InterventionEstimator::copy(
            const ASISTStudy3InterventionEstimator& estimator) {
            Estimator::copy(estimator);

            this->last_time_step = estimator.last_time_step;

            //            this->room_id_to_idx = estimator.room_id_to_idx;
            //            this->room_ids = estimator.room_ids;
            //            this->threat_room_belief_estimators =
            //                estimator.threat_room_belief_estimators;
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
            if (this->containers_initialized) {
                this->last_areas =
                    vector<vector<string>>(new_data.get_num_data_points());
                this->last_placed_markers =
                    vector<vector<Marker>>(new_data.get_num_data_points());
                this->active_unspoken_markers =
                    vector<vector<Marker>>(new_data.get_num_data_points());

                for (int d = 0; d < new_data.get_num_data_points(); d++) {
                    this->last_areas[d] = vector<string>(3);
                    this->last_placed_markers[d] = vector<Marker>(3);
                    this->active_unspoken_markers[d] = vector<Marker>(3);
                }
            }
        }

        void ASISTStudy3InterventionEstimator::estimate_motivation(
            const EvidenceSet& new_data) {
            const auto& metadata = new_data.get_metadata();

            check_field(metadata[0], "mission_order");

            auto& encouragement_node =
                dynamic_pointer_cast<ASISTStudy3InterventionModel>(this->model)
                    ->get_encouragement_node();

            Eigen::VectorXd increments =
                Eigen::VectorXd::Zero(new_data.get_num_data_points());
            if (metadata[0]["mission_order"] == 1) {
                for (int d = 0; d < new_data.get_num_data_points(); d++) {
                    for (int t = 0; t < new_data.get_time_steps(); t++) {
                        increments[d] +=
                            (int)new_data
                                .get_dict_like_data()[d][t]
                                                     [Labels::ENCOURAGEMENT];
                    }
                }

                encouragement_node->increment_assignment(increments);
            }
            else {
                if (encouragement_node->get_size() == 0) {
                    // The agent crashed and data from mission 1 was lost.
                    // Assume there was no encouragement utterances to force
                    // intervention.
                    encouragement_node->set_assignment(increments);
                }
            }
        }

        void ASISTStudy3InterventionEstimator::estimate_unspoken_markers(
            const EvidenceSet& new_data) {

            for (int d = 0; d < new_data.get_num_data_points(); d++) {
                for (int t = 0; t < new_data.get_time_steps(); t++) {
                    for (int player_order = 0; player_order < 3;
                         player_order++) {

                        const auto& unspoken_marker =
                            this->active_unspoken_markers[d][player_order];
                        if (unspoken_marker.type != MarkerType::NONE) {
                            if (did_player_speak_about_marker(player_order,
                                                              unspoken_marker,
                                                              d,
                                                              t,
                                                              new_data)) {
                                this->clear_active_unspoken_marker(player_order,
                                                                   d);
                            }
                        }

                        // Check if a new marker was placed and needs to be
                        // monitored for communication.
                        Marker new_marker = get_last_placed_marker(
                            player_order, d, t, new_data);
                        const auto& last_marker =
                            this->last_placed_markers[d][player_order];
                        if (new_marker.is_none()) {
                            bool cond1 = did_player_change_area(
                                player_order, d, t, new_data);
                            bool cond2 = did_player_interact_with_victim(
                                player_order, d, t, new_data);
                            bool cond3 =
                                !last_marker.is_none() &&
                                is_player_far_apart(player_order,
                                                    last_marker.position,
                                                    MAX_DIST,
                                                    d,
                                                    t,
                                                    new_data);
                            if (cond1 || cond2 || cond3) {
                                this->active_unspoken_markers[d][player_order] =
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
                                this->active_unspoken_markers[d][player_order] =
                                    last_marker;
                            }

                            this->last_placed_markers[d][player_order] =
                                new_marker;
                        }
                    }
                }
            }
        }

        bool ASISTStudy3InterventionEstimator::did_player_change_area(
            int player_order,
            int data_point,
            int time_step,
            const EvidenceSet& new_data) {

            const string& curr_area =
                new_data.get_dict_like_data()[data_point][time_step]
                                             [Labels::AREA][player_order];
            const string& last_area =
                this->last_areas[data_point][player_order];

            return curr_area != last_area;
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
                json_info["encouragement_cdf"] =
                    to_string(this->encouragement_cdf);
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

        Eigen::VectorXd
        ASISTStudy3InterventionEstimator::get_encouragement_cdfs() {
            auto& encouragement_node =
                dynamic_pointer_cast<ASISTStudy3InterventionModel>(this->model)
                    ->get_encouragement_node();
            this->encouragement_cdf = encouragement_node->get_cdfs(1, 0, false);
            return this->encouragement_cdf;
        }

        void ASISTStudy3InterventionEstimator::clear_active_unspoken_marker(
            int player_order, int data_point) {
            this->active_unspoken_markers[data_point][player_order] = Marker();
        }

        //----------------------------------------------------------------------
        // Getters & Setters
        //----------------------------------------------------------------------

        int ASISTStudy3InterventionEstimator::get_last_time_step() const {
            return last_time_step;
        }

        const vector<vector<ASISTStudy3InterventionEstimator::Marker>>&
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
