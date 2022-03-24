#include "ASISTStudy3InterventionEstimator.h"

#include <fmt/format.h>

#include "asist/study3/ASISTStudy3InterventionModel.h"
#include "asist/study3/ASISTStudy3MessageConverter.h"
#include "utils/JSONChecker.h"

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors
        //----------------------------------------------------------------------
        ASISTStudy3InterventionEstimator::ASISTStudy3InterventionEstimator(
            const shared_ptr<ASISTStudy3InterventionModel>& model)
            : Estimator(model) {}

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

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        void ASISTStudy3InterventionEstimator::copy(
            const ASISTStudy3InterventionEstimator& estimator) {
            Estimator::copy(estimator);

            //            this->room_id_to_idx = estimator.room_id_to_idx;
            //            this->room_ids = estimator.room_ids;
            //            this->threat_room_belief_estimators =
            //                estimator.threat_room_belief_estimators;
        }

        void ASISTStudy3InterventionEstimator::estimate(
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
                        const string& label =
                            ASISTStudy3MessageConverter::Labels::ENCOURAGEMENT;
                        increments[d] +=
                            (int)new_data.get_dict_like_data()[d][t][label];
                    }
                }

                encouragement_node->increment_assignment(increments);
            }
        }

        void
        ASISTStudy3InterventionEstimator::get_info(nlohmann::json& json) const {
            json["name"] = this->get_name();
            json["encouragement_cdf"] = this->encouragement_cdf;
        }

        string ASISTStudy3InterventionEstimator::get_name() const {
            return NAME;
        }

        void ASISTStudy3InterventionEstimator::prepare() {
            //            // belief about threat rooms
            //            for (auto& estimators :
            //            this->threat_room_belief_estimators) {
            //                for (auto& estimator : estimators) {
            //                    estimator.prepare();
            //                }
            //            }
        }

        Eigen::VectorXd
        ASISTStudy3InterventionEstimator::get_encouragement_cdfs() {
            auto& encouragement_node =
                dynamic_pointer_cast<ASISTStudy3InterventionModel>(this->model)
                    ->get_encouragement_node();
            this->encouragement_cdf = encouragement_node->get_pdfs(1, 0)(0);
            return encouragement_node->get_pdfs(1, 0);
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
