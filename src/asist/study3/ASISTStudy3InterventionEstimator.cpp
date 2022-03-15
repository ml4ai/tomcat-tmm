#include "ASISTStudy3InterventionEstimator.h"

#include <fmt/format.h>

#include "pgm/DynamicBayesNet.h"

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors
        //----------------------------------------------------------------------
        ASISTStudy3InterventionEstimator::InterventionEstimator(
            const string& map_filepath,
            const string& threat_room_model_filepath) {

            this->parse_map(map_filepath);
            this->create_belief_estimators(threat_room_model_filepath);
        }

        //----------------------------------------------------------------------
        // Copy & Move constructors/assignments
        //----------------------------------------------------------------------
        ASISTStudy3InterventionEstimator::InterventionEstimator(
            const ASISTStudy3InterventionEstimator& estimator) {
            this->copy_estimator(estimator);
        }

        ASISTStudy3InterventionEstimator&
        ASISTStudy3InterventionEstimator::operator=(
            const ASISTStudy3InterventionEstimator& estimator) {
            this->copy_estimator(estimator);
            return *this;
        }

        //----------------------------------------------------------------------
        // Static functions
        //----------------------------------------------------------------------

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        void ASISTStudy3InterventionEstimator::copy(const ASISTStudy3InterventionEstimator& estimator) {
            Estimator::copy_estimator(estimator);
            this->room_id_to_idx = estimator.room_id_to_idx;
            this->room_ids = estimator.room_ids;
            this->threat_room_belief_estimators =
                estimator.threat_room_belief_estimators;
        }

        void ASISTStudy3InterventionEstimator::estimate(const EvidenceSet& new_data) {
            this->update_threat_room_beliefs(new_data);
        }

        void
        ASISTStudy3InterventionEstimator::get_info(nlohmann::json& json) const {
            json["name"] = this->get_name();
            json["inference_horizon"] = this->inference_horizon;
        }

        string ASISTStudy3InterventionEstimator::get_name() const { return "sampler"; }

        bool ASISTStudy3InterventionEstimator::is_binary_on_prediction() const {
            // Irrelevant for this estimator. It just aggregates other smaller
            // estimators. One per belief.
            return false;
        }

        void ASISTStudy3InterventionEstimator::prepare() {
            // belief about threat rooms
            for (auto& estimators : this->threat_room_belief_estimators) {
                for (auto& estimator : estimators) {
                    estimator.prepare();
                }
            }
        }

        void ASISTStudy3InterventionEstimator::parse_map(const std::string& map_filepath) {
            fstream map_file;
            map_file.open(map_filepath);
            if (map_file.is_open()) {
                nlohmann::json json_map = nlohmann::json::parse(map_file);

                for (const auto& location : json_map["locations"]) {
                    const string& type = location["type"];
                    const string& id = location["id"];

                    if (type == "room") {
                        this->room_id_to_idx[id] = this->room_ids.size();
                        this->room_ids.push_back(id);
                    }
                }
            }
            else {
                throw TomcatModelException(
                    fmt::format("File {} could not be open.", map_filepath));
            }
        }

        void ASISTStudy3InterventionEstimator::create_belief_estimators(
            const string& threat_room_model_filepath) {
            this->create_threat_room_belief_estimators(
                threat_room_model_filepath);
        }

        void
        ASISTStudy3InterventionEstimator::create_threat_room_belief_estimators(
            const string& threat_room_model_filepath) {

            // Since we are using SumProduct, we can use the same model for all
            // estimators since the SP estimator does not uses the model
            // directly but a factor graph created from the model.
            DBNPtr belief_model = make_shared<DynamicBayesNet>(
                DynamicBayesNet::create_from_json(threat_room_model_filepath));
            belief_model->unroll(3, true);

            this->threat_room_belief_estimators =
                vector<vector<SumProductEstimator>>(NUM_ROLES);
            for (int i = 0; i < NUM_ROLES; i++) {
                for (int j = 0; j < this->room_ids.size(); j++) {
                    // Instantiate a new belief model.
                    SumProductEstimator estimator(belief_model, 0, "Belief");
                    this->threat_room_belief_estimators[i].push_back(estimator);
                }
            }
        }

        void ASISTStudy3InterventionEstimator::update_threat_room_beliefs(
            const EvidenceSet& new_data) {
            for (int i = 0; i < NUM_ROLES; i++) {
                for (int j = 0; j < this->room_ids.size(); j++) {
                    EvidenceSet threat_room_data;
                    string original_label =
                        fmt::format("ThreatMarkerInRoom#{}", this->room_ids[j]);
                    threat_room_data.add_data("ThreatMarkerInRoom",
                                              new_data[original_label]);
                    original_label = fmt::format("ThreatMarkerInFoVInRoom#{}",
                                                 this->room_ids[j]);
                    threat_room_data.add_data("ThreatMarkerInFoV",
                                              new_data[original_label]);
                    original_label = fmt::format("SpeechAboutThreatRoom#{}",
                                                 this->room_ids[j]);
                    threat_room_data.add_data("SpeechAboutThreat",
                                              new_data[original_label]);

                    this->threat_room_belief_estimators[i][j].estimate(
                        threat_room_data);
                }
            }
        }

        //----------------------------------------------------------------------
        // Getters & Setters
        //----------------------------------------------------------------------

    } // namespace model
} // namespace tomcat
