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
        nlohmann::json ASISTStudy2EstimateReporter::get_header_section(
            const AgentPtr& agent) const {
            nlohmann::json header;
            header["timestamp"] = this->get_current_timestamp();
            header["message_type"] = "agent";
            header["version"] = agent->get_version();

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

            nlohmann::json msg_common;
            nlohmann::json msg_state;
            nlohmann::json msg_prediction;

            // Common fields
            msg_common["created"] =
                this->get_timestamp_at(agent, time_step, data_point);
            msg_common["group"]["start"] = nullptr;
            msg_common["group"]["duration"] = nullptr;
            msg_common["group"]["explanation"]["M1"] =
                "The final score is estimated by simulating actions until the "
                "end of the mission based on a Markov process. The average "
                "number of actions consistent with regular and critical victim "
                "rescues per player are used to estimate the most likely score "
                "to be achieved by the team.";
            msg_common["group"]["explanation"]["M3"] =
                "The map version assignment is estimated by the most likely "
                "visible section per player based on transitions from one "
                "section to another and how long the players spend on a given "
                "area of the building. Different planning conditions and role "
                "affect how the player favors areas of the map and are also "
                "take into consideration int the inference. The final "
                "assignment is given by the most likely valid combination of "
                "independent visible sections per player.";
            msg_common["group"]["explanation"]["M6"] =
                "The legend version is estimated based a Markov process over "
                "player's belief about two main actions: rescuing victims in a "
                "room or leaving victims in a room. Marker placements, victim "
                "rescues, and player's marker legend version are some of the "
                "variables that help to update this belief. The change of "
                "marker meaning is also taking into consideration as dependent "
                "Markov chain that gets updated throughout the mission.";
            msg_common["group"]["explanation"]["M7"] =
                "The next action taken in face of a marker block is estimated "
                "by generating the most probable scenarios 5 time steps ahead. "
                "States consistent with the player being out of a marker block "
                "range of detection are taken into consideration to estimate "
                "the most probable area the player ends up in. A player's "
                "intent follows a Markov process getting updated to reflect "
                "the player's disposition to enter or leaving rooms for a "
                "given marker by a given player at different times during the "
                "mission. The player marker legend version works as a prior on "
                "the player's intent.";
            msg_common["group"]["explanation"]["Reasoning Engine"] =
                "The agent uses a probabilistic framework in all of the "
                "predictions.";
            msg_common["group"]["explanation"]["Granularity"] =
                "Beliefs are updated at every second.";
            msg_common["predictions"] = nlohmann::json::array();

            msg_state = msg_common;
            msg_prediction = msg_common;

            for (const auto& estimator : agent->get_estimators()) {
                for (const auto& base_estimator :
                     estimator->get_base_estimators()) {

                    if (const auto& score_estimator =
                            dynamic_pointer_cast<FinalTeamScoreEstimator>(
                                base_estimator)) {
                        this->add_final_team_score_prediction(
                            msg_state["predictions"],
                            agent,
                            score_estimator,
                            time_step,
                            data_point);
                    }
                    else if (const auto& next_action_estimator =
                                 dynamic_pointer_cast<
                                     NextAreaOnNearbyMarkerEstimator>(
                                     base_estimator)) {
                        this->add_marker_false_belief_prediction(
                            msg_prediction["predictions"],
                            agent,
                            next_action_estimator,
                            time_step,
                            data_point);
                    }
                    else if (const auto& map_estimator = dynamic_pointer_cast<
                                 IndependentMapVersionAssignmentEstimator>(
                                 base_estimator)) {
                        this->add_map_info_prediction(msg_state["predictions"],
                                                      agent,
                                                      map_estimator,
                                                      time_step,
                                                      data_point);
                    }
                    else if (
                        const auto& marker_estimator = dynamic_pointer_cast<
                            IndependentMarkerLegendVersionAssignmentEstimator>(
                            base_estimator)) {
                        this->add_marker_legend_prediction(
                            msg_state["predictions"],
                            agent,
                            marker_estimator,
                            time_step,
                            data_point);
                    }
                }
            }

            if (msg_state["predictions"].empty())
                msg_state.clear();
            if (msg_prediction["predictions"].empty())
                msg_prediction.clear();

            return {msg_state, msg_prediction};
        }

        void ASISTStudy2EstimateReporter::add_final_team_score_prediction(
            nlohmann::json& json_predictions,
            const AgentPtr& agent,
            const shared_ptr<FinalTeamScoreEstimator>& estimator,
            int time_step,
            int data_point) const {

            if (time_step == 240 || time_step == 540 || time_step == 840) {
                double score = estimator->get_estimates().estimates[0](
                    data_point, time_step);
                double avg_regular = estimator->get_estimates().custom_data[0](
                    data_point, time_step);
                double std_regular = estimator->get_estimates().custom_data[1](
                    data_point, time_step);
                double avg_critical = estimator->get_estimates().custom_data[2](
                    data_point, time_step);
                double std_critical = estimator->get_estimates().custom_data[3](
                    data_point, time_step);

                nlohmann::json prediction;
                boost::uuids::uuid u = boost::uuids::random_generator()();
                prediction["unique_id"] = boost::uuids::to_string(u);
                prediction["start"] =
                    this->get_timestamp_at(agent, time_step, data_point);
                prediction["duration"] =
                    agent->get_evidence_metadata()[data_point]["step_size"];
                prediction["subject"] =
                    agent->get_evidence_metadata()[data_point]["team_id"];
                prediction["predicted_property"] = "Score";
                prediction["prediction"] = score;
                prediction["probability_type"] = "";
                prediction["probability"] = "";
                prediction["confidence_type"] = "";
                prediction["confidence"] = "";
                prediction["explanation"]["regular_victim"]
                          ["avg_number_rescues"] = avg_regular;
                prediction["explanation"]["regular_victim"]
                          ["std_number_rescues"] = std_regular;
                prediction["explanation"]["critical_victim"]
                          ["avg_number_rescues"] = avg_regular;
                prediction["explanation"]["critical_victim"]
                          ["avg_number_rescues"] = std_regular;

                json_predictions.push_back(prediction);
            }
        }

        void ASISTStudy2EstimateReporter::add_map_info_prediction(
            nlohmann::json& json_predictions,
            const AgentPtr& agent,
            const shared_ptr<IndependentMapVersionAssignmentEstimator>&
                estimator,
            int time_step,
            int data_point) const {

            if (time_step == 120 || time_step == 420 || time_step == 720) {
                nlohmann::json prediction;
                boost::uuids::uuid u = boost::uuids::random_generator()();
                prediction["unique_id"] = boost::uuids::to_string(u);
                prediction["start"] =
                    this->get_timestamp_at(agent, time_step, data_point);
                prediction["duration"] =
                    agent->get_evidence_metadata()[data_point]["step_size"];

                double probability = 0;
                int assignment = 0;
                for (const auto& probabilities :
                     estimator->get_estimates().estimates) {
                    if (probabilities(data_point, time_step) > probability) {
                        probability = probabilities(data_point, time_step);
                        assignment++;
                    }
                }

                prediction["predicted_property"] = "Map Version";
                prediction["probability_type"] = "float";
                prediction["probability"] = probability;
                prediction["confidence_type"] = "";
                prediction["confidence"] = "";
                prediction["explanation"] = nullptr;

                vector<string> map_assignment;
                switch (assignment) {
                case 0:
                    map_assignment = {"SaturnA_24", "SaturnA_34", "SaturnA_64"};
                    break;

                case 1:
                    map_assignment = {"SaturnA_34", "SaturnA_24", "SaturnA_64"};
                    break;

                case 2:
                    map_assignment = {"SaturnA_34", "SaturnA_24", "SaturnA_64"};
                    break;

                case 3:
                    map_assignment = {"SaturnA_34", "SaturnA_64", "SaturnA_24"};
                    break;

                case 4:
                    map_assignment = {"SaturnA_64", "SaturnA_24", "SaturnA_34"};
                    break;

                case 5:
                    map_assignment = {"SaturnA_64", "SaturnA_34", "SaturnA_24"};
                    break;
                }

                prediction["prediction"] = nlohmann::json::array();
                int i = 0;
                for (const auto& json_player :
                     agent->get_evidence_metadata()[data_point]["players"]) {
                    nlohmann::json marker_assignment;
                    marker_assignment["participant_id"] = json_player["id"];
                    marker_assignment["map"] = map_assignment[i];
                    prediction["prediction"].push_back(marker_assignment);
                    i++;
                }

                json_predictions.push_back(prediction);
            }
        }

        void ASISTStudy2EstimateReporter::add_marker_legend_prediction(
            nlohmann::json& json_predictions,
            const AgentPtr& agent,
            const shared_ptr<IndependentMarkerLegendVersionAssignmentEstimator>&
                estimator,
            int time_step,
            int data_point) const {

            if (time_step == 180 || time_step == 480 || time_step == 780) {
                nlohmann::json prediction;
                boost::uuids::uuid u = boost::uuids::random_generator()();
                prediction["unique_id"] = boost::uuids::to_string(u);
                prediction["start"] =
                    this->get_timestamp_at(agent, time_step, data_point);
                prediction["duration"] =
                    agent->get_evidence_metadata()[data_point]["step_size"];

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
                prediction["explanation"] = nullptr;

                int i = 0;
                for (const auto& json_player :
                     agent->get_evidence_metadata()[data_point]["players"]) {
                    nlohmann::json marker_assignment;
                    marker_assignment["participant_id"] = json_player["id"];
                    marker_assignment["markerblocks"] =
                        assignment == i ? "B_Sally" : "A_Anne";
                    prediction["prediction"].push_back(marker_assignment);
                    i++;
                }
                json_predictions.push_back(prediction);
            }
        }

        void ASISTStudy2EstimateReporter::add_marker_false_belief_prediction(
            nlohmann::json& json_predictions,
            const AgentPtr& agent,
            const shared_ptr<NextAreaOnNearbyMarkerEstimator>& estimator,
            int time_step,
            int data_point) const {

            if (this->m7_initialized) {
                this->initialize_m7(agent->get_evidence_metadata());
            }

            if (EXISTS(time_step,
                       this->m7_time_steps_per_data_point[data_point])) {

                nlohmann::json prediction;
                boost::uuids::uuid u = boost::uuids::random_generator()();
                prediction["unique_id"] = boost::uuids::to_string(u);
                prediction["start"] =
                    this->get_timestamp_at(agent, time_step, data_point);
                prediction["duration"] =
                    agent->get_evidence_metadata()[data_point]["step_size"];
                prediction["probability_type"] = "float";
                prediction["confidence_type"] = "";
                prediction["confidence"] = "";

                int i = 0;
                for (const auto& json_m7_event :
                     this->m7_time_steps_per_data_point[data_point]
                                                       [time_step]) {

                    int subject_number = json_m7_event["subject_number"];
                    int placed_by_number = json_m7_event["placed_by_number"];

                    if (estimator->get_player_number() != subject_number ||
                        estimator->get_placed_by_player_nummber() !=
                            placed_by_number)
                        continue;

                    double probability_hallway =
                        estimator->get_estimates().estimates[0](data_point,
                                                                time_step);
                    if (probability_hallway >= 0.5) {
                        prediction["action"] = "will not enter room";
                        prediction["probability"] = probability_hallway;
                    }
                    else {
                        prediction["action"] = "will enter room";
                        prediction["probability"] = 1 - probability_hallway;
                    }

                    prediction["subject"] = json_m7_event["subject"];
                    nlohmann::json json_marker;
                    json_marker["location"]["x"] = json_m7_event["marker_x"];
                    json_marker["location"]["z"] = json_m7_event["marker_z"];
                    json_marker["color"] = json_m7_event["placed_by"];
                    prediction["using"] =
                        json_marker == 1 ? "MARKER_BLOCK_1" : "MARKER_BLOCK_2";
                    prediction["object"] = json_m7_event["door_id"];

                    double prop_valid_scenarios =
                        estimator->get_estimates().custom_data[0](data_point,
                                                                  time_step);
                    json_predictions.push_back(prediction);
                    i++;
                }
            }
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
                    };
                }
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

    } // namespace model
} // namespace tomcat
