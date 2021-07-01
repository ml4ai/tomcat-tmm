#include "ASISTStudy2EstimateReporter.h"

#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <fmt/format.h>

#include "converter/ASISTMultiPlayerMessageConverter.h"
#include "pipeline/estimation/custom_metrics/FinalTeamScoreEstimator.h"

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        ASISTStudy2EstimateReporter::ASISTStudy2EstimateReporter() {}

        ASISTStudy2EstimateReporter::~ASISTStudy2EstimateReporter() {}

        //----------------------------------------------------------------------
        // Copy & Move constructors/assignments
        //----------------------------------------------------------------------
        ASISTStudy2EstimateReporter::ASISTStudy2EstimateReporter(
            const ASISTStudy2EstimateReporter& reporter) {}

        ASISTStudy2EstimateReporter& ASISTStudy2EstimateReporter::operator=(
            const ASISTStudy2EstimateReporter& reporter) {
            return *this;
        }

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        nlohmann::json ASISTStudy2EstimateReporter::get_header_section(
            const AgentPtr& agent) const {
            nlohmann::json header;
            header["timestamp"] = this->get_current_timestamp();
            header["message_type"] = "agent";
            header["version"] = "1.0";

            return header;
        }

        pair<nlohmann::json, nlohmann::json>
        ASISTStudy2EstimateReporter::get_msg_section(const AgentPtr& agent,
                                                     int data_point) const {
            nlohmann::json msg_state;
            nlohmann::json msg_prediction;
            msg_state["trial_id"] =
                agent->get_evidence_metadata()[data_point]["trial_id"];
            msg_state["experiment_id"] =
                agent->get_evidence_metadata()[data_point]["experiment_id"];
            msg_state["timestamp"] = this->get_current_timestamp();
            msg_state["source"] = agent->get_id();
            msg_state["sub_type"] = "prediction:state";
            msg_state["version"] = "1.0";

            msg_prediction = msg_state;
            msg_prediction["sub_type"] = "prediction:action";

            return {msg_state, msg_prediction};
        }

        pair<nlohmann::json, nlohmann::json>
        ASISTStudy2EstimateReporter::get_data_section(const AgentPtr& agent,
                                                      int time_step,
                                                      int data_point) const {

            nlohmann::json msg_state;
            nlohmann::json msg_prediction;

            const string& initial_timestamp =
                agent->get_evidence_metadata()[data_point]["initial_timestamp"];
            int elapsed_time =
                time_step *
                (int)agent->get_evidence_metadata()[data_point]["step_size"];

            // Common fields
            msg_state["created"] =
                this->get_elapsed_timestamp(initial_timestamp, elapsed_time);
            msg_state["predictions"] = nlohmann::json::array();
            msg_prediction = msg_state;

            for (const auto& estimator : agent->get_estimators()) {
                for (const auto& base_estimator :
                     estimator->get_base_estimators()) {

                    if (base_estimator->get_estimates().label ==
                            FinalTeamScoreEstimator::LABEL) {
                        this->add_final_team_score_prediction(
                            msg_state["predictions"],
                            agent,
                            base_estimator,
                            time_step,
                            data_point);
                    }
                    else if (base_estimator->get_estimates().label ==
                                 FinalTeamScoreEstimator::LABEL) {
                        this->add_marker_false_belief_prediction(
                            msg_prediction["predictions"],
                            agent,
                            base_estimator,
                            time_step,
                            data_point);
                    }
                    else if (base_estimator->get_estimates().label.rfind(
                                 ASISTMultiPlayerMessageConverter::
                                     MAP_INFO_ASSIGNMENT,
                                 0)) {
                        this->add_map_info_prediction(msg_state["predictions"],
                                                      agent,
                                                      base_estimator,
                                                      time_step,
                                                      data_point);
                    }
                    else if (base_estimator->get_estimates().label.rfind(
                                 ASISTMultiPlayerMessageConverter::
                                     MARKER_LEGEND_ASSIGNMENT,
                                 0)) {
                        this->add_marker_legend_prediction(
                            msg_state["predictions"],
                            agent,
                            base_estimator,
                            time_step,
                            data_point);
                    }
                }
            }

            return {msg_state, msg_prediction};
        }

        void ASISTStudy2EstimateReporter::add_final_team_score_prediction(
            nlohmann::json& json_predictions,
            const AgentPtr& agent,
            const EstimatorPtr& estimator,
            int time_step,
            int data_point) const {

            double score =
                estimator->get_estimates().estimates[0](data_point, time_step);

            nlohmann::json prediction;
            prediction["unique_id"] =
                boost::uuids::to_string(boost::uuids::uuid());
            prediction["subject"] =
                agent->get_evidence_metadata()[data_point]["team_id"];
            prediction["predicted_property"] = "Score";
            prediction["prediction"] = score;
            prediction["probability_type"] = "";
            prediction["probability"] = "";
            prediction["confidence_type"] = "";
            prediction["confidence"] = "";
            prediction["explanation"] =
                "Calculated by the number of predicted regular and critical "
                "victim rescues until the end of the mission, estimated by "
                "sampling future PlayerState variables (using the transition "
                "distribution) given observations up to the current "
                "time.";

            json_predictions["predictions"].push_back(prediction);
        }

        void ASISTStudy2EstimateReporter::add_map_info_prediction(
            nlohmann::json& json_predictions,
            const AgentPtr& agent,
            const EstimatorPtr& estimator,
            int time_step,
            int data_point) const {

            nlohmann::json prediction;
            prediction["unique_id"] =
                boost::uuids::to_string(boost::uuids::uuid());

            double probability = 0;
            int assignment = 0;
            for (const auto& probabilities :
                 estimator->get_estimates().estimates) {
                if (probabilities(data_point, time_step) > probability) {
                    probability = probabilities(data_point, time_step);
                    assignment++;
                }
            }

            prediction["prediction"] = nlohmann::json::array();
            prediction["predicted_property"] = "Map Version";
            prediction["probability_type"] = "float";
            prediction["probability"] = probability;
            prediction["confidence_type"] = "";
            prediction["confidence"] = "";
            prediction["explanation"] = fmt::format(
                "Assignment of the variable {} (which determines which "
                "player was assigned to which map) with highest inferred "
                "probability.",
                ASISTMultiPlayerMessageConverter::MAP_INFO_ASSIGNMENT);

            vector<string> map_assignment;
            switch (assignment) {
            case 0:
                map_assignment = {"SaturnA_14", "SaturnA_15", "SaturnA_16"};
                break;

            case 1:
                map_assignment = {"SaturnA_14", "SaturnA_16", "SaturnA_15"};
                break;

            case 2:
                map_assignment = {"SaturnA_15", "SaturnA_14", "SaturnA_16"};
                break;

            case 3:
                map_assignment = {"SaturnA_15", "SaturnA_16", "SaturnA_14"};
                break;

            case 4:
                map_assignment = {"SaturnA_16", "SaturnA_14", "SaturnA_15"};
                break;

            case 5:
                map_assignment = {"SaturnA_16", "SaturnA_15", "SaturnA_14"};
                break;
            }

            int i = 0;
            for (const auto& json_player :
                 agent->get_evidence_metadata()[data_point]["players"]) {
                nlohmann::json marker_assignment;
                marker_assignment["participant_id"] = json_player["id"];
                marker_assignment["map"] = map_assignment[i];
                prediction["prediction"].push_back(marker_assignment);
                i++;
            }
            json_predictions["predictions"].push_back(prediction);
        }

        void ASISTStudy2EstimateReporter::add_marker_legend_prediction(
            nlohmann::json& json_predictions,
            const AgentPtr& agent,
            const EstimatorPtr& estimator,
            int time_step,
            int data_point) const {

            nlohmann::json prediction;
            prediction["unique_id"] =
                boost::uuids::to_string(boost::uuids::uuid());

            double probability = 0;
            int assignment = 0;
            for (const auto& probabilities :
                 estimator->get_estimates().estimates) {
                if (probabilities(data_point, time_step) > probability) {
                    probability = probabilities(data_point, time_step);
                    assignment++;
                }
            }

            prediction["prediction"] = nlohmann::json::array();
            prediction["predicted_property"] = "Marker Legend";
            prediction["probability_type"] = "float";
            prediction["probability"] = probability;
            prediction["confidence_type"] = "";
            prediction["confidence"] = "";
            prediction["explanation"] = fmt::format(
                "Assignment of the variable {} "
                "(which determines which player was assigned to which marker "
                "legend version) with highest inferred probability.",
                ASISTMultiPlayerMessageConverter::MARKER_LEGEND_ASSIGNMENT);

            int i = 0;
            for (const auto& json_player :
                 agent->get_evidence_metadata()[data_point]["players"]) {
                nlohmann::json marker_assignment;
                marker_assignment["participant_id"] = json_player["id"];
                marker_assignment["markerblocks"] = assignment == i ? "B" : "A";
                prediction["prediction"].push_back(marker_assignment);
                i++;
            }
            json_predictions["predictions"].push_back(prediction);
        }

        void ASISTStudy2EstimateReporter::add_marker_false_belief_prediction(
            nlohmann::json& json_predictions,
            const AgentPtr& agent,
            const EstimatorPtr& estimator,
            int time_step,
            int data_point) const {

            nlohmann::json prediction;
            prediction["unique_id"] =
                boost::uuids::to_string(boost::uuids::uuid());

            prediction["predicted_property"] = "False Belief";
            prediction["probability_type"] = "float";
            prediction["confidence_type"] = "";
            prediction["confidence"] = "";

            int i = 0;
            for (const auto& json_player :
                agent->get_evidence_metadata()[data_point]["players"]) {

                // Get it by player
                double probability = 0.4;
                int assignment = 0;

                prediction["action"] =
                    assignment == 0 ? "will not enter room" : "will enter room";
                prediction["subject"] = json_player["id"];
                prediction["probability"] = probability;

                // TODO - get from data
                nlohmann::json json_marker;
                json_marker["location"]["x"] = 1000;
                json_marker["location"]["z"] = 2000;
                json_marker["color"] = "Green";
                json_marker["label"] = "A";
                prediction["using"][""] = json_marker;

                // TODO - closest door
                prediction["object"] = 1;

                int block_seen = 1;
                int marker_legend = 0;
                if (block_seen == 1) {
                    if (marker_legend == 0) {
                        prediction["explanation"] =
                            "The probability of Intent be EnterRoom is larger "
                            "than StayInHallway, given that the player saw a "
                            "block 1 and received marker legend info A.";
                    }
                    else {
                        prediction["explanation"] =
                            "The probability of Intent be StayInHallway is "
                            "larger than EnterRoom, given that the player saw "
                            "a block 1 and received marker legend info B.";
                    }
                }
                else {
                    if (marker_legend == 0) {
                        prediction["explanation"] =
                            "The probability of Intent be StayInHallway is "
                            "larger than EnterRoom, given that the player saw "
                            "a block 2 and received marker legend info A.";
                    }
                    else {
                        prediction["explanation"] =
                            "The probability of Intent be EnterRoom is larger "
                            "than StayInHallway, given that the player saw a "
                            "block 2 and received marker legend info B.";
                    }
                }

                json_predictions["predictions"].push_back(prediction);
                i++;
            }
        }

    } // namespace model
} // namespace tomcat
