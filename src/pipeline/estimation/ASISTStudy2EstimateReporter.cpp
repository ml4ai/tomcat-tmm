#include "ASISTStudy2EstimateReporter.h"

#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <fmt/format.h>

#include "converter/ASISTMultiPlayerMessageConverter.h"

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

        vector<nlohmann::json>
        ASISTStudy2EstimateReporter::translate_estimates_to_messages(
            const AgentPtr& agent, int time_step) const {
            vector<nlohmann::json> messages;

            nlohmann::json state_message;
            nlohmann::json prediction_message;
            state_message["header"] = this->get_header_section(agent);
            prediction_message["header"] = this->get_header_section(agent);

            int num_data_points = agent->get_evidence_metadata().size();

            for (int d = 0; d < num_data_points; d++) {
                state_message["msg"] = this->get_common_msg_section(agent, d);
                state_message["msg"]["sub_type"] = "Prediction:State";

                prediction_message["msg"] =
                    this->get_common_msg_section(agent, d);
                prediction_message["msg"]["sub_type"] = "Prediction:Action";

                vector<nlohmann::json> predictions;
                for (const auto& estimator : agent->get_estimators()) {
                    for (const auto& base_estimator :
                         estimator->get_base_estimators()) {
                        if (const auto& score_estimator =
                                dynamic_pointer_cast<FinalTeamScoreEstimator>(
                                    base_estimator)) {

                            predictions =
                                this->get_final_team_score_predictions(
                                    agent, score_estimator, time_step, d);

                            for (const auto& prediction : predictions) {
                                state_message["data"] = prediction["data"];
                                state_message["data"]["group"]["explanation"] =
                                    "The final score is estimated by "
                                    "simulating "
                                    "actions until the end of the mission "
                                    "based on "
                                    "a "
                                    "Markov process. The average number of "
                                    "actions "
                                    "consistent with regular and critical "
                                    "victim "
                                    "rescues per player is used to estimate "
                                    "the "
                                    "most "
                                    "likely score to be achieved by the team.";
                                messages.push_back(state_message);
                            }
                        }
                        else if (base_estimator->get_estimates().label ==
                                 "MapVersionAssignment") {

                            predictions = this->get_map_info_predictions(
                                agent, base_estimator, time_step, d);

                            for (const auto& prediction : predictions) {
                                state_message["data"] = prediction["data"];
                                state_message["data"]["group"]["explanation"] =
                                    "The map version assignment is estimated "
                                    "by looking at the areas of the map "
                                    "accessed by the players. We hypothesize "
                                    "that these areas are determined by the "
                                    "players role and map version they have "
                                    "access to. Given observed areas and "
                                    "roles, the map version is estimated by "
                                    "doing inference over the described "
                                    "generative model.";
                                messages.push_back(state_message);
                            }
                        }
                        else if (base_estimator->get_estimates().label ==
                                 "MarkerLegendVersionAssignment") {

                            predictions = this->get_marker_legend_predictions(
                                agent, base_estimator, time_step, d);

                            for (const auto& prediction : predictions) {
                                state_message["data"] = prediction["data"];
                                state_message["data"]["group"]["explanation"] =
                                    "The legend version is estimated based on "
                                    "the victim's perception by the players "
                                    "and the agent's belief about the meaning "
                                    "the team adopted for the markers.";
                                messages.push_back(state_message);
                            }
                        }
                        else if (const auto& next_action_estimator =
                                     dynamic_pointer_cast<
                                         NextAreaOnNearbyMarkerEstimator>(
                                         base_estimator)) {

                            predictions = this->get_next_area_predictions(
                                agent, next_action_estimator, time_step, d);

                            for (const auto& prediction : predictions) {
                                prediction_message["data"] = prediction["data"];
                                prediction_message["data"]["group"]["explanatio"
                                                                    "n"] =
                                    "The next action taken in face of a marker "
                                    "block "
                                    "is estimated by computing the probability "
                                    "that the player will be in a room in the "
                                    "next second after leaving the range of a "
                                    "marker block. "
                                    "The model keeps two Markov "
                                    "chains: one representing the player's "
                                    "intent and another one representing his "
                                    "belief about the meaning of the markers "
                                    "over time. Markers and next areas are "
                                    "observations emitted from the player's "
                                    "intent.";
                                messages.push_back(prediction_message);
                            }
                        }
                    }
                }
            }

            return messages;
        }

        nlohmann::json ASISTStudy2EstimateReporter::build_log_message(
            const AgentPtr& agent, const string& log) const {
            // No predefined format for the ASIST program
            nlohmann::json message;
            return message;
        }

        nlohmann::json ASISTStudy2EstimateReporter::get_header_section(
            const AgentPtr& agent) const {
            nlohmann::json header;
            header["timestamp"] = this->get_current_timestamp();
            header["message_type"] = "agent";
            header["version"] = agent->get_version();

            return header;
        }

        nlohmann::json ASISTStudy2EstimateReporter::get_common_msg_section(
            const AgentPtr& agent, int data_point) const {
            nlohmann::json msg_common;
            msg_common["trial_id"] =
                agent->get_evidence_metadata()[data_point]["trial_unique_id"];
            msg_common["experiment_id"] =
                agent->get_evidence_metadata()[data_point]["experiment_id"];
            msg_common["timestamp"] = this->get_current_timestamp();
            msg_common["source"] = agent->get_id();
            msg_common["version"] = "1.0";
            msg_common["trial_number"] =
                agent->get_evidence_metadata()[data_point]["trial_id"];

            return msg_common;
        }

        nlohmann::json ASISTStudy2EstimateReporter::get_common_data_section(
            const AgentPtr& agent, int data_point) const {
            nlohmann::json msg_common;
            msg_common["group"]["start_elapsed_time"] = nullptr;
            msg_common["group"]["duration"] =
                agent->get_evidence_metadata()[data_point]["step_size"];
            ;
            msg_common["predictions"] = nlohmann::json::array();

            return msg_common;
        }

        vector<nlohmann::json>
        ASISTStudy2EstimateReporter::get_final_team_score_predictions(
            const AgentPtr& agent,
            const shared_ptr<FinalTeamScoreEstimator>& estimator,
            int time_step,
            int data_point) const {

            vector<nlohmann::json> predictions;

            if (time_step == NO_OBS) {
                // Batch processing
                for (int t : {240, 540, 840}) {
                    nlohmann::json json_data =
                        this->get_common_data_section(agent, data_point);
                    json_data["data"]["created_elapsed_time"] =
                        get_milliseconds_at(agent, t, data_point);

                    nlohmann::json json_predictions = nlohmann::json::array();
                    double score =
                        estimator->get_estimates().estimates[0](data_point, t);
                    double avg_regular =
                        estimator->get_estimates().custom_data[0](data_point,
                                                                  t);
                    double std_regular =
                        estimator->get_estimates().custom_data[1](data_point,
                                                                  t);
                    double avg_critical =
                        estimator->get_estimates().custom_data[2](data_point,
                                                                  t);
                    double std_critical =
                        estimator->get_estimates().custom_data[3](data_point,
                                                                  t);

                    nlohmann::json json_prediction;
                    boost::uuids::uuid u = boost::uuids::random_generator()();
                    json_prediction["unique_id"] = boost::uuids::to_string(u);
                    json_prediction["start_elapsed_time"] =
                        this->get_milliseconds_at(agent, t, data_point);
                    json_prediction["duration"] =
                        agent->get_evidence_metadata()[data_point]["step_size"];
                    json_prediction["subject_type"] = "team";
                    json_prediction["subject"] =
                        agent->get_evidence_metadata()[data_point]["team_id"];
                    json_prediction["predicted_property"] =
                        "M1:team_performance";
                    json_prediction["prediction"] = score;
                    json_prediction["probability_type"] = "";
                    json_prediction["probability"] = "";
                    json_prediction["confidence_type"] = "";
                    json_prediction["confidence"] = "";
                    json_prediction["explanation"]["regular_victim"]
                                   ["avg_number_rescues"] = avg_regular;
                    json_prediction["explanation"]["regular_victim"]
                                   ["std_number_rescues"] = std_regular;
                    json_prediction["explanation"]["critical_victim"]
                                   ["avg_number_rescues"] = avg_critical;
                    json_prediction["explanation"]["critical_victim"]
                                   ["std_number_rescues"] = std_critical;
                    json_predictions.push_back(json_prediction);

                    json_data["data"]["predictions"] = json_predictions;
                    predictions.push_back(json_data);
                }
            }
            else {
                // Online processing
            }

            return predictions;
        }

        vector<nlohmann::json>
        ASISTStudy2EstimateReporter::get_map_info_predictions(
            const AgentPtr& agent,
            const EstimatorPtr& estimator,
            int time_step,
            int data_point) const {

            vector<nlohmann::json> predictions;

            if (time_step == NO_OBS) {
                // Batch processing
                for (int t : {120, 420, 720}) {
                    nlohmann::json json_data =
                        this->get_common_data_section(agent, data_point);
                    json_data["data"]["created_elapsed_time"] =
                        get_milliseconds_at(agent, t, data_point);

                    double probability = -1;
                    int assignment = -1;
                    for (int i = 0;
                         i < estimator->get_estimates().estimates.size();
                         i++) {
                        double temp = estimator->get_estimates().estimates[i](
                            data_point, t);
                        if (temp > probability) {
                            probability = temp;
                            assignment = i;
                        }
                    }

                    vector<string> map_assignment;
                    switch (assignment) {
                    case 0:
                        map_assignment = {
                            "SaturnA_24", "SaturnA_34", "SaturnA_64"};
                        break;

                    case 1:
                        map_assignment = {
                            "SaturnA_24", "SaturnA_64", "SaturnA_34"};
                        break;

                    case 2:
                        map_assignment = {
                            "SaturnA_34", "SaturnA_24", "SaturnA_64"};
                        break;

                    case 3:
                        map_assignment = {
                            "SaturnA_34", "SaturnA_64", "SaturnA_24"};
                        break;

                    case 4:
                        map_assignment = {
                            "SaturnA_64", "SaturnA_24", "SaturnA_34"};
                        break;

                    case 5:
                        map_assignment = {
                            "SaturnA_64", "SaturnA_34", "SaturnA_24"};
                        break;
                    }

                    nlohmann::json json_predictions = nlohmann::json::array();
                    int i = 0;
                    for (const auto& json_player :
                         agent
                             ->get_evidence_metadata()[data_point]["players"]) {
                        nlohmann::json prediction;
                        boost::uuids::uuid u =
                            boost::uuids::random_generator()();
                        prediction["unique_id"] = boost::uuids::to_string(u);
                        prediction["start_elapsed_time"] =
                            this->get_milliseconds_at(agent, t, data_point);
                        prediction["duration"] =
                            agent->get_evidence_metadata()[data_point]
                                                          ["step_size"];
                        prediction["subject_type"] = "individual";
                        prediction["subject"] = json_player["id"];
                        prediction["predicted_property"] = "M3:participant_map";
                        prediction["prediction"] = map_assignment[i];
                        prediction["probability_type"] = "float";
                        prediction["probability"] = probability;
                        prediction["confidence_type"] = "";
                        prediction["confidence"] = "";
                        prediction["explanation"] = nullptr;
                        json_predictions.push_back(prediction);
                        i++;
                    }

                    json_data["data"]["predictions"] = json_predictions;
                    predictions.push_back(json_data);
                }
            }
            else {
                // Online processing
            }

            return predictions;
        }

        vector<nlohmann::json>
        ASISTStudy2EstimateReporter::get_marker_legend_predictions(
            const AgentPtr& agent,
            const EstimatorPtr& estimator,
            int time_step,
            int data_point) const {

            vector<nlohmann::json> predictions;

            if (time_step == NO_OBS) {
                // Batch processing
                for (int t : {180, 480, 780}) {
                    nlohmann::json json_data =
                        this->get_common_data_section(agent, data_point);
                    json_data["data"]["created_elapsed_time"] =
                        get_milliseconds_at(agent, t, data_point);

                    double probability = -1;
                    int assignment = -1;
                    for (int i = 0;
                         i < estimator->get_estimates().estimates.size();
                         i++) {
                        double temp = estimator->get_estimates().estimates[i](
                            data_point, t);
                        if (temp > probability) {
                            probability = temp;
                            assignment = i;
                        }
                    }

                    nlohmann::json json_predictions = nlohmann::json::array();
                    int i = 0;
                    for (const auto& json_player :
                         agent
                             ->get_evidence_metadata()[data_point]["players"]) {
                        nlohmann::json prediction;
                        boost::uuids::uuid u =
                            boost::uuids::random_generator()();
                        prediction["unique_id"] = boost::uuids::to_string(u);
                        prediction["start_elapsed_time"] =
                            this->get_milliseconds_at(agent, t, data_point);
                        prediction["duration"] =
                            agent->get_evidence_metadata()[data_point]
                                                          ["step_size"];
                        prediction["subject_type"] = "individual";
                        prediction["subject"] = json_player["id"];
                        prediction["predicted_property"] =
                            "M6:participant_block_legend";
                        prediction["prediction"] =
                            assignment == i ? "B_Sally" : "A_Anne";
                        prediction["probability_type"] = "float";
                        prediction["probability"] = probability;
                        prediction["confidence_type"] = "";
                        prediction["confidence"] = "";
                        prediction["explanation"] = nullptr;
                        json_predictions.push_back(prediction);
                        i++;
                    }

                    json_data["data"]["predictions"] = json_predictions;
                    predictions.push_back(json_data);
                }
            }
            else {
                // Online processing
            }

            return predictions;
        }

        vector<nlohmann::json>
        ASISTStudy2EstimateReporter::get_next_area_predictions(
            const AgentPtr& agent,
            const shared_ptr<NextAreaOnNearbyMarkerEstimator>& estimator,
            int time_step,
            int data_point) const {

            vector<nlohmann::json> predictions;

            if (time_step == NO_OBS) {
                // Batch processing
                const auto& json_events =
                    agent->get_evidence_metadata()[data_point]["m7_events"];

                Eigen::MatrixXd hallway_probabilities =
                    estimator->get_estimates().estimates[0];

                for (const auto& json_m7_event : json_events) {
                    if (json_m7_event["marker_number"] !=
                        estimator->get_marker_number()) {
                        continue;
                    }

                    int subject_number = json_m7_event["subject_number"];
                    int placed_by_number = json_m7_event["placed_by_number"];

                    if (estimator->get_player_number() != subject_number ||
                        estimator->get_placed_by_player_nummber() !=
                            placed_by_number)
                        continue;

                    int t = json_m7_event["time_step"];

                    nlohmann::json json_data =
                        this->get_common_data_section(agent, data_point);
                    json_data["data"]["created_elapsed_time"] =
                        json_m7_event["start_elapsed_time"];

                    nlohmann::json json_predictions = nlohmann::json::array();
                    nlohmann::json prediction;
                    boost::uuids::uuid u = boost::uuids::random_generator()();
                    prediction["unique_id"] = boost::uuids::to_string(u);
                    prediction["duration"] = 1;
                    prediction["predicted_property"] = "M7:room_enter";
                    prediction["probability_type"] = "float";
                    prediction["confidence_type"] = "";
                    prediction["confidence"] = "";

                    prediction["start_elapsed_time"] =
                        json_m7_event["start_elapsed_time"];

                    double probability_hallway =
                        hallway_probabilities(data_point, t);
                    if (probability_hallway > 0.5) {
                        prediction["action"] = "will_not_enter_room";
                        prediction["probability"] = probability_hallway;
                    }
                    else {
                        prediction["action"] = "will_enter_room";
                        prediction["probability"] = 1 - probability_hallway;
                    }

                    nlohmann::json json_using;
                    json_using["location"]["x"] = json_m7_event["marker_x"];
                    json_using["location"]["y"] = 60;
                    json_using["location"]["z"] = json_m7_event["marker_z"];
                    json_using["callsign"] = json_m7_event["placed_by"];
                    json_using["type"] = json_m7_event["marker_number"] == 1
                                             ? "Marker Block 1"
                                             : "Marker Block 2";
                    prediction["using"] = json_using;
                    prediction["subject"] = json_m7_event["subject_id"];
                    prediction["object"] = json_m7_event["door_id"];

                    double prop_valid_scenarios =
                        estimator->get_estimates().custom_data[0](data_point,
                                                                  t);
                    prediction["explanation"]["prop_valid_scenarios"] =
                        prop_valid_scenarios;

                    json_predictions.push_back(prediction);

                    json_data["data"]["predictions"] = json_predictions;
                    predictions.push_back(json_data);
                }
            }
            else {
                // Online processing
            }

            return predictions;
        }

        void ASISTStudy2EstimateReporter::initialize_m7(
            const nlohmann::json& evidence_metadata) const {
            this->m7_time_steps_per_data_point =
                vector<unordered_map<int, vector<nlohmann::json>>>(
                    evidence_metadata.size());
            int d = 0;
            for (const auto& json_data_point : evidence_metadata) {
                unordered_map<int, vector<nlohmann::json>>
                    events_info_per_time_step;
                for (const auto& json_m7_event : json_data_point["m7_events"]) {
                    int time_step = json_m7_event["time_step"];
                    if (EXISTS(time_step, events_info_per_time_step)) {
                        events_info_per_time_step[time_step].push_back(
                            json_m7_event);
                    }
                    else {
                        events_info_per_time_step[time_step] = {json_m7_event};
                    }
                }
                this->m7_time_steps_per_data_point[d] =
                    events_info_per_time_step;
                d++;
            }
            this->m7_initialized = true;
        }

        string ASISTStudy2EstimateReporter::get_timestamp_at(
            const AgentPtr& agent, int time_step, int data_point) const {
            const string& initial_timestamp =
                agent->get_evidence_metadata()[data_point]["initial_timestamp"];
            int elapsed_time =
                time_step *
                (int)agent->get_evidence_metadata()[data_point]["step_size"];

            return this->get_elapsed_timestamp(initial_timestamp, elapsed_time);
        }

        int ASISTStudy2EstimateReporter::get_milliseconds_at(
            const AgentPtr& agent, int time_step, int data_point) const {
            int step_size =
                agent->get_evidence_metadata()[data_point]["step_size"];
            return time_step * step_size * 1000;
        }

    } // namespace model
} // namespace tomcat
