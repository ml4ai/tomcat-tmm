#include <fstream>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/replace.hpp>
#include <boost/program_options.hpp>
#include <nlohmann/json.hpp>

#include "converter/ASISTMultiPlayerMessageConverter.h"
#include "pgm/EvidenceSet.h"
#include "pipeline/estimation/custom_metrics/NextAreaOnNearbyMarkerEstimator.h"
#include "utils/FileHandler.h"

using namespace tomcat::model;
using namespace std;
namespace po = boost::program_options;
#define add_player_suffix(label, player_num)                                   \
    ASISTMultiPlayerMessageConverter::get_player_variable_label(label,         \
                                                                player_num)
#define NO_MARKER ASISTMultiPlayerMessageConverter::NO_NEARBY_MARKER
#define NO_PLANNING ASISTMultiPlayerMessageConverter::NO_TEAM_PLANNING
#define NO_PLANNING ASISTMultiPlayerMessageConverter::NO_TEAM_PLANNING
#define MARKER_LEGEND_A ASISTMultiPlayerMessageConverter::MARKER_LEGEND_A
#define MARKER_LEGEND_B ASISTMultiPlayerMessageConverter::MARKER_LEGEND_B
#define HALLWAY ASISTMultiPlayerMessageConverter::HALLWAY
#define ROOM ASISTMultiPlayerMessageConverter::HALLWAY
#define MARKER_AREA 2

#define P1_NEARBY_MARKER_LABEL                                                 \
    ASISTMultiPlayerMessageConverter::PLAYER1_NEARBY_MARKER_LABEL
#define P2_NEARBY_MARKER_LABEL                                                 \
    ASISTMultiPlayerMessageConverter::PLAYER2_NEARBY_MARKER_LABEL
#define P3_NEARBY_MARKER_LABEL                                                 \
    ASISTMultiPlayerMessageConverter::PLAYER3_NEARBY_MARKER_LABEL
#define PLAYER_AREA_LABEL ASISTMultiPlayerMessageConverter::PLAYER_AREA_LABEL
#define MARKER_LEGEND_ASSIGNMENT_LABEL                                         \
    ASISTMultiPlayerMessageConverter::MARKER_LEGEND_ASSIGNMENT_LABEL
#define PLAYER_MARKER_LEGEND_VERSION_LABEL                                     \
    ASISTMultiPlayerMessageConverter::PLAYER_MARKER_LEGEND_VERSION_LABEL

#define NEXT_AREA_AFTER_P1_MARKER_LABEL "NextAreaAfterP1Marker"
#define NEXT_AREA_AFTER_P2_MARKER_LABEL "NextAreaAfterP2Marker"
#define NEXT_AREA_AFTER_P3_MARKER_LABEL "NextAreaAfterP3Marker"
#define NEXT_AREA_AFTER_P1_MARKER1_LABEL "NextAreaAfterP1Marker1"
#define NEXT_AREA_AFTER_P2_MARKER1_LABEL "NextAreaAfterP2Marker1"
#define NEXT_AREA_AFTER_P3_MARKER1_LABEL "NextAreaAfterP3Marker1"
#define NEXT_AREA_AFTER_P1_MARKER2_LABEL "NextAreaAfterP1Marker2"
#define NEXT_AREA_AFTER_P2_MARKER2_LABEL "NextAreaAfterP2Marker2"
#define NEXT_AREA_AFTER_P3_MARKER2_LABEL "NextAreaAfterP3Marker2"
#define P1_NEARBY_MARKER1_LABEL "Player1NearbyMarker1"
#define P2_NEARBY_MARKER1_LABEL "Player2NearbyMarker1"
#define P3_NEARBY_MARKER1_LABEL "Player3NearbyMarker1"
#define P1_NEARBY_MARKER2_LABEL "Player1NearbyMarker2"
#define P2_NEARBY_MARKER2_LABEL "Player2NearbyMarker2"
#define P3_NEARBY_MARKER2_LABEL "Player3NearbyMarker2"
#define P1_MARKER_RANGE_LABEL "Player1PlayerMarkerRange"
#define P2_MARKER_RANGE_LABEL "Player2PlayerMarkerRange"
#define P3_MARKER_RANGE_LABEL "Player3PlayerMarkerRange"

const inline string PLAYER1_PLAYER_MARKER_AREA_LABEL =
    "Player1PlayerMarkerArea";
const inline string PLAYER2_PLAYER_MARKER_AREA_LABEL =
    "Player2PlayerMarkerArea";
const inline string PLAYER3_PLAYER_MARKER_AREA_LABEL =
    "Player3PlayerMarkerArea";

enum Event { within_range, out_of_range };

void create_trial_period_data(const string& input_dir,
                              const string& output_dir,
                              int num_periods) {
    EvidenceSet data(input_dir);

    if (num_periods > 1 && data.get_time_steps() % num_periods == 0) {
        Eigen::MatrixXd periods(data.get_num_data_points(),
                                data.get_time_steps());

        int num_cols_per_period = data.get_time_steps() / num_periods;
        for (int p = 0; p < num_periods; p++) {
            periods.block(0,
                          p * num_cols_per_period,
                          data.get_num_data_points(),
                          num_cols_per_period) =
                Eigen::MatrixXd::Constant(
                    data.get_num_data_points(), num_cols_per_period, p);
        }

        EvidenceSet trial_period_data;
        string label = "TrialPeriod" + to_string(num_periods);
        trial_period_data.add_data(label, periods);
        trial_period_data.save(output_dir);
    }
}

/**
 * Checks if the trial is the first (no planning) or the second one and the
 * planning condition is true.
 */
void create_planning_before_trial_data(const string& input_dir,
                                       const string& output_dir) {
    EvidenceSet data(input_dir);

    if (!data.get_metadata().empty()) {
        Eigen::MatrixXd planning_before_trial(data.get_num_data_points(),
                                              data.get_time_steps());
        Eigen::MatrixXd trial_order(data.get_num_data_points(),
                                    data.get_time_steps());

        for (int d = 0; d < data.get_num_data_points(); d++) {
            const auto& json_file = data.get_metadata()[d];
            int trial_number = stoi(((string)json_file["trial_id"]).substr(1));

            if (trial_number % 2 == 0) {
                int planning = data[ASISTMultiPlayerMessageConverter::
                                        PLANNING_CONDITION_LABEL]
                                   .at(0, d, 0);
                planning_before_trial.row(d) =
                    Eigen::VectorXd::Constant(data.get_time_steps(), planning);
                trial_order.row(d) =
                    Eigen::VectorXd::Constant(data.get_time_steps(), 1);
            }
            else {
                // There's never a planning session before the first trial of a
                // team
                planning_before_trial.row(d) = Eigen::VectorXd::Constant(
                    data.get_time_steps(), NO_PLANNING);
                trial_order.row(d) =
                    Eigen::VectorXd::Constant(data.get_time_steps(), 0);
            }
        }

        EvidenceSet new_data;
        new_data.add_data("PlanningBeforeTrial", planning_before_trial);
        new_data.add_data("TrialOrder", trial_order);
        new_data.save(output_dir);
    }
}

void create_m7_data_from_external_source(const string& input_dir,
                                         const string& output_dir,
                                         const string& external_filepath) {
    struct M7_Event {
        long start_elapsed_time;
        long resolved_elapsed_time;
        string subject_id;
        string door_id;
        int marker_number;
        int marker_x;
        int marker_z;
        string marker_placed_by;
        int next_area;
    };

    struct M7_Data {
        string subject_with_legend_b;
        vector<M7_Event> events;
    };

    // Indexes of the columns with the corresponding data
    const int TRIAL_IDX = 1;
    const int SUBJECT_IDX = 2;
    const int MEASURE_IDX = 3;
    const int GROUND_TRUTH_IDX = 4;
    const int DOOR_IDX = 5;
    const int START_ELAPSED_TIME_IDX = 6;
    const int RESOLVED_ELAPSED_TIME_IDX = 7;
    const int MARKER_X_IDX = 8;
    const int MARKER_Z_IDX = 9;
    const int MARKER_PLACED_BY_IDX = 10;
    const int MARKER_TYPE_IDX = 11;

    unordered_map<string, M7_Data> trial_to_data;
    ifstream file(external_filepath);
    if (file.good()) {
        string line;
        getline(file, line); // Skip header
        while (!file.eof()) {
            getline(file, line);

            vector<string> tokens;
            boost::split(tokens, line, boost::is_any_of(","));
            const string& measure = tokens[MEASURE_IDX];

            if (measure != "M6" && measure != "M7")
                continue;

            // Some entries contain TMXXXX_T000XX. This strips the trial id from
            // the token no matter the format
            string trial_id =
                tokens[TRIAL_IDX].substr(tokens[TRIAL_IDX].find("T0"));

            if (!EXISTS(trial_id, trial_to_data)) {
                trial_to_data[trial_id] = {};
            }

            auto& m7_data = trial_to_data[trial_id];
            if (measure == "M6") {
                if (tokens[GROUND_TRUTH_IDX][0] == 'B') {
                    m7_data.subject_with_legend_b = tokens[SUBJECT_IDX];
                }
            }
            else if (measure == "M7") {
                M7_Event m7_event;
                m7_event.start_elapsed_time =
                    stoi(tokens[START_ELAPSED_TIME_IDX]);
                m7_event.resolved_elapsed_time =
                    stoi(tokens[RESOLVED_ELAPSED_TIME_IDX]);

                if (m7_event.start_elapsed_time == NO_OBS) {
                    cout << trial_id << ": Invalid start elapsed time.\n";
                    continue;
                }

                if (tokens[MARKER_TYPE_IDX].find("1") != string::npos) {
                    m7_event.marker_number = 1;
                }
                else if (tokens[MARKER_TYPE_IDX].find("2") != string::npos) {
                    m7_event.marker_number = 2;
                }
                else {
                    continue;
                }

                m7_event.subject_id = tokens[SUBJECT_IDX];
                m7_event.door_id = tokens[DOOR_IDX];

                m7_event.marker_x = stoi(tokens[MARKER_X_IDX]);
                m7_event.marker_z = stoi(tokens[MARKER_Z_IDX]);
                m7_event.marker_placed_by = tokens[MARKER_PLACED_BY_IDX];
                if (tokens[GROUND_TRUTH_IDX] != "False" &&
                    tokens[GROUND_TRUTH_IDX] != "True") {
                    m7_event.next_area = NO_OBS;
                }
                else {
                    m7_event.next_area =
                        (int)(tokens[GROUND_TRUTH_IDX] == "True");
                }
                m7_data.events.push_back(m7_event);
            }
        }
    }

    // Updates observations for nearby markers, player area, next area,
    // legend version and legend assignment from the information in the file.
    // This overrides keeps previous events but adds new ones if not already
    // present. If ground truth data is available, player areas might change to
    // reflect the ground truth if they are wrong.
    EvidenceSet data(input_dir);
    if (!data.get_metadata().empty()) {
        int num_rows = data.get_num_data_points();
        int num_cols = data.get_time_steps();

        Eigen::MatrixXd marker_legend_assignment =
            Eigen::MatrixXd::Constant(num_rows, num_cols, NO_OBS);
        vector<Eigen::MatrixXd> marker_legend_version_per_player(
            3, Eigen::MatrixXd::Constant(num_rows, num_cols, MARKER_LEGEND_A));

        vector<Eigen::MatrixXd> next_area_per_player1(
            3, Eigen::MatrixXd::Constant(num_rows, num_cols, NO_OBS));
        vector<Eigen::MatrixXd> next_area_per_player2(
            3, Eigen::MatrixXd::Constant(num_rows, num_cols, NO_OBS));
        vector<Eigen::MatrixXd> nearby_marker_per_player1(
            3, Eigen::MatrixXd::Constant(num_rows, num_cols, NO_MARKER));
        vector<Eigen::MatrixXd> nearby_marker_per_player2(
            3, Eigen::MatrixXd::Constant(num_rows, num_cols, NO_MARKER));

        // Per marker. Binary variable.
        vector<Eigen::MatrixXd> nearby_marker1_per_player1(
            3, Eigen::MatrixXd::Constant(num_rows, num_cols, NO_MARKER));
        vector<Eigen::MatrixXd> nearby_marker1_per_player2(
            3, Eigen::MatrixXd::Constant(num_rows, num_cols, NO_MARKER));
        vector<Eigen::MatrixXd> nearby_marker2_per_player1(
            3, Eigen::MatrixXd::Constant(num_rows, num_cols, NO_MARKER));
        vector<Eigen::MatrixXd> nearby_marker2_per_player2(
            3, Eigen::MatrixXd::Constant(num_rows, num_cols, NO_MARKER));

        nlohmann::json json_new_metadata = data.get_metadata();

        for (int d = 0; d < num_rows; d++) {
            nlohmann::json& json_file = json_new_metadata[d];
            const auto& m7_data = trial_to_data[json_file["trial_id"]];

            unordered_map<string, int> player_callsign_to_number(3);
            unordered_map<string, int> player_id_to_number(3);
            int player_number = 0;
            for (const auto& json_player : json_file["players"]) {
                player_callsign_to_number[json_player["callsign"]] =
                    player_number;
                player_id_to_number[json_player["id"]] = player_number;
                player_number++;
            }

            // Marker legend version
            player_number = player_id_to_number[m7_data.subject_with_legend_b];
            marker_legend_version_per_player[player_number].row(d) =
                Eigen::VectorXd::Constant(num_cols, MARKER_LEGEND_B);

            // Marker legend version assignment
            marker_legend_assignment.row(d) =
                Eigen::VectorXd::Constant(num_cols, player_number);

            json_file["m7_events"] = nlohmann::json::array();
            for (const auto& m7_event : m7_data.events) {
                player_number = player_id_to_number[m7_event.subject_id];
                int other_player_number =
                    player_callsign_to_number[m7_event.marker_placed_by];

                int start_elapsed_milliseconds =
                    m7_event.start_elapsed_time -
                    (int)json_file["initial_elapsed_milliseconds"];
                int resolved_elapsed_milliseconds =
                    m7_event.resolved_elapsed_time -
                    (int)json_file["initial_elapsed_milliseconds"];
                int time_step =
                    min(start_elapsed_milliseconds / 1000, num_cols - 1);

                // Include the new event
                int t = time_step;
                if ((player_number == 0 && other_player_number == 1) ||
                    (player_number == 1 && other_player_number == 0) ||
                    (player_number == 2 && other_player_number == 0)) {

                    nearby_marker_per_player1[player_number](d, t) =
                        m7_event.marker_number;
                    next_area_per_player1[player_number](d, t) =
                        m7_event.next_area;

                    if (m7_event.marker_number == 1) {
                        nearby_marker1_per_player1[player_number](d, t) = 1;
                    }
                    else {
                        nearby_marker2_per_player1[player_number](d, t) = 1;
                    }
                }
                else {
                    nearby_marker_per_player2[player_number](d, t) =
                        m7_event.marker_number;
                    next_area_per_player2[player_number](d, t) =
                        m7_event.next_area;

                    if (m7_event.marker_number == 1) {
                        nearby_marker1_per_player2[player_number](d, t) = 1;
                    }
                    else {
                        nearby_marker2_per_player2[player_number](d, t) = 1;
                    }
                }

                // Update metadata. Information in this file will be used
                // in the report generation
                nlohmann::json json_m7_event;
                json_m7_event["time_step"] = time_step;
                json_m7_event["start_elapsed_time"] =
                    m7_event.start_elapsed_time;
                json_m7_event["resolved_elapsed_time"] =
                    m7_event.resolved_elapsed_time;
                json_m7_event["marker_number"] = m7_event.marker_number;
                json_m7_event["marker_x"] = m7_event.marker_x;
                json_m7_event["marker_z"] = m7_event.marker_z;
                json_m7_event["door_id"] = m7_event.door_id;
                json_m7_event["subject_id"] = m7_event.subject_id;
                json_m7_event["placed_by"] = m7_event.marker_placed_by;
                json_m7_event["subject_number"] = player_number;
                json_m7_event["placed_by_number"] = other_player_number;

                json_file["m7_events"].push_back(json_m7_event);
            }
        }

        // Create new dataset. Replace metadata and some observations.
        EvidenceSet new_data = data;
        new_data.set_metadata(json_new_metadata);

        new_data.add_data(MARKER_LEGEND_ASSIGNMENT_LABEL,
                          marker_legend_assignment);
        new_data.add_data(
            add_player_suffix(PLAYER_MARKER_LEGEND_VERSION_LABEL, 1),
            marker_legend_version_per_player[0]);
        new_data.add_data(
            add_player_suffix(PLAYER_MARKER_LEGEND_VERSION_LABEL, 2),
            marker_legend_version_per_player[1]);
        new_data.add_data(
            add_player_suffix(PLAYER_MARKER_LEGEND_VERSION_LABEL, 3),
            marker_legend_version_per_player[2]);

        new_data.add_data(add_player_suffix(NEXT_AREA_AFTER_P2_MARKER_LABEL, 1),
                          next_area_per_player1[0]);
        new_data.add_data(add_player_suffix(NEXT_AREA_AFTER_P3_MARKER_LABEL, 1),
                          next_area_per_player2[0]);
        new_data.add_data(add_player_suffix(NEXT_AREA_AFTER_P1_MARKER_LABEL, 2),
                          next_area_per_player1[1]);
        new_data.add_data(add_player_suffix(NEXT_AREA_AFTER_P3_MARKER_LABEL, 2),
                          next_area_per_player2[1]);
        new_data.add_data(add_player_suffix(NEXT_AREA_AFTER_P1_MARKER_LABEL, 3),
                          next_area_per_player1[2]);
        new_data.add_data(add_player_suffix(NEXT_AREA_AFTER_P2_MARKER_LABEL, 3),
                          next_area_per_player2[2]);

        new_data.add_data(add_player_suffix(P2_NEARBY_MARKER_LABEL, 1),
                          nearby_marker_per_player1[0]);
        new_data.add_data(add_player_suffix(P3_NEARBY_MARKER_LABEL, 1),
                          nearby_marker_per_player2[0]);
        new_data.add_data(add_player_suffix(P1_NEARBY_MARKER_LABEL, 2),
                          nearby_marker_per_player1[1]);
        new_data.add_data(add_player_suffix(P3_NEARBY_MARKER_LABEL, 2),
                          nearby_marker_per_player2[1]);
        new_data.add_data(add_player_suffix(P1_NEARBY_MARKER_LABEL, 3),
                          nearby_marker_per_player1[2]);
        new_data.add_data(add_player_suffix(P2_NEARBY_MARKER_LABEL, 3),
                          nearby_marker_per_player2[2]);

        new_data.add_data(add_player_suffix(P2_NEARBY_MARKER1_LABEL, 1),
                          nearby_marker1_per_player1[0]);
        new_data.add_data(add_player_suffix(P3_NEARBY_MARKER1_LABEL, 1),
                          nearby_marker1_per_player2[0]);
        new_data.add_data(add_player_suffix(P1_NEARBY_MARKER1_LABEL, 2),
                          nearby_marker1_per_player1[1]);
        new_data.add_data(add_player_suffix(P3_NEARBY_MARKER1_LABEL, 2),
                          nearby_marker1_per_player2[1]);
        new_data.add_data(add_player_suffix(P1_NEARBY_MARKER1_LABEL, 3),
                          nearby_marker1_per_player1[2]);
        new_data.add_data(add_player_suffix(P2_NEARBY_MARKER1_LABEL, 3),
                          nearby_marker1_per_player2[2]);

        new_data.add_data(add_player_suffix(P2_NEARBY_MARKER2_LABEL, 1),
                          nearby_marker2_per_player1[0]);
        new_data.add_data(add_player_suffix(P3_NEARBY_MARKER2_LABEL, 1),
                          nearby_marker2_per_player2[0]);
        new_data.add_data(add_player_suffix(P1_NEARBY_MARKER2_LABEL, 2),
                          nearby_marker2_per_player1[1]);
        new_data.add_data(add_player_suffix(P3_NEARBY_MARKER2_LABEL, 2),
                          nearby_marker2_per_player2[1]);
        new_data.add_data(add_player_suffix(P1_NEARBY_MARKER2_LABEL, 3),
                          nearby_marker2_per_player1[2]);
        new_data.add_data(add_player_suffix(P2_NEARBY_MARKER2_LABEL, 3),
                          nearby_marker2_per_player2[2]);

        new_data.save(output_dir);
    }
}

void create_m7_player_areas(const string& input_dir, const string& output_dir) {
    EvidenceSet data(input_dir);

    const auto& area_p1 = data[add_player_suffix(PLAYER_AREA_LABEL, 1)];
    const auto& area_p2 = data[add_player_suffix(PLAYER_AREA_LABEL, 2)];
    const auto& area_p3 = data[add_player_suffix(PLAYER_AREA_LABEL, 3)];

    data.add_data(add_player_suffix("Player2" + PLAYER_AREA_LABEL, 1), area_p1);
    data.add_data(add_player_suffix("Player3" + PLAYER_AREA_LABEL, 1), area_p1);
    data.add_data(add_player_suffix("Player1" + PLAYER_AREA_LABEL, 2), area_p2);
    data.add_data(add_player_suffix("Player3" + PLAYER_AREA_LABEL, 2), area_p2);
    data.add_data(add_player_suffix("Player1" + PLAYER_AREA_LABEL, 3), area_p3);
    data.add_data(add_player_suffix("Player2" + PLAYER_AREA_LABEL, 3), area_p3);

    data.save(output_dir);
}

void create_observations_for_next_accessed_area(const string& input_dir,
                                                const string& output_dir) {
    EvidenceSet data(input_dir);

    if (!data.get_metadata().empty()) {
        int num_rows = data.get_num_data_points();
        int num_cols = data.get_time_steps();

        vector<Eigen::MatrixXd> next_area_per_player1(
            3, Eigen::MatrixXd::Constant(num_rows, num_cols, NO_OBS));
        vector<Eigen::MatrixXd> next_area_per_player2(
            3, Eigen::MatrixXd::Constant(num_rows, num_cols, NO_OBS));

        vector<Eigen::MatrixXd> next_area_per_player1_m1(
            3, Eigen::MatrixXd::Constant(num_rows, num_cols, NO_OBS));
        vector<Eigen::MatrixXd> next_area_per_player2_m1(
            3, Eigen::MatrixXd::Constant(num_rows, num_cols, NO_OBS));
        vector<Eigen::MatrixXd> next_area_per_player1_m2(
            3, Eigen::MatrixXd::Constant(num_rows, num_cols, NO_OBS));
        vector<Eigen::MatrixXd> next_area_per_player2_m2(
            3, Eigen::MatrixXd::Constant(num_rows, num_cols, NO_OBS));

        vector<Eigen::MatrixXd> marker_range_player1(
            3, Eigen::MatrixXd::Zero(num_rows, num_cols));
        vector<Eigen::MatrixXd> marker_range_player2(
            3, Eigen::MatrixXd::Zero(num_rows, num_cols));

        for (int player_number = 0; player_number < 3; player_number++) {
            Eigen::MatrixXd nearby_markers_by_1;
            Eigen::MatrixXd nearby_markers_by_2;

            if (player_number == 0) {
                nearby_markers_by_1 = data[add_player_suffix(
                    P2_NEARBY_MARKER_LABEL, player_number + 1)](0, 0);
                nearby_markers_by_2 = data[add_player_suffix(
                    P3_NEARBY_MARKER_LABEL, player_number + 1)](0, 0);
            }
            else if (player_number == 1) {
                nearby_markers_by_1 = data[add_player_suffix(
                    P1_NEARBY_MARKER_LABEL, player_number + 1)](0, 0);
                nearby_markers_by_2 = data[add_player_suffix(
                    P3_NEARBY_MARKER_LABEL, player_number + 1)](0, 0);
            }
            else {
                nearby_markers_by_1 = data[add_player_suffix(
                    P1_NEARBY_MARKER_LABEL, player_number + 1)](0, 0);
                nearby_markers_by_2 = data[add_player_suffix(
                    P2_NEARBY_MARKER_LABEL, player_number + 1)](0, 0);
            }

            for (int d = 0; d < num_rows; d++) {
                Event event1 = Event::out_of_range;
                Event event2 = Event::out_of_range;
                int time_event1;
                int time_event2;
                int marker_event1;
                int marker_event2;
                for (int t = 0; t < num_cols; t++) {
                    // First pair of players
                    if (event1 == Event::out_of_range &&
                        nearby_markers_by_1(d, t) != NO_MARKER) {
                        // Player enters the range of a marker block;
                        event1 = Event::within_range;
                        time_event1 = t;
                        marker_event1 = nearby_markers_by_1(d, t);
                    }
                    else if (event1 == Event::within_range &&
                             nearby_markers_by_1(d, t) == NO_MARKER) {
                        // Player leaves the range of a marker block;
                        event1 = Event::out_of_range;

                        // Fill the resolution area for all time steps the
                        // player spends on a set of markers. This will be used
                        // to evaluate the performance of the model.
                        int current_area =
                            data[add_player_suffix(PLAYER_AREA_LABEL,
                                                   player_number + 1)]
                                .at(0, d, t);

                        for (int i = time_event1; i < t; i++) {
                            next_area_per_player1[player_number](d, i) =
                                current_area;
                        }

                        // Unique observation when transition occurs. This will
                        // be used as a variable in the model.
                        if (marker_event1 == 1) {
                            next_area_per_player1_m1[player_number](d, t) =
                                current_area;
                        }
                        else {
                            next_area_per_player1_m2[player_number](d, t) =
                                current_area;
                        }
                    }

                    if (event1 == Event::within_range) {
                        marker_range_player1[player_number](d, t) = 1;
                    }

                    // Second pair of players
                    if (event2 == Event::out_of_range &&
                        nearby_markers_by_2(d, t) != NO_MARKER) {
                        // Player enters the range of a marker block;
                        event2 = Event::within_range;
                        time_event2 = t;
                        marker_event2 = nearby_markers_by_2(d, t);
                    }
                    else if (event2 == Event::within_range &&
                             nearby_markers_by_2(d, t) == NO_MARKER) {
                        // Player leaves the range of a marker block;
                        event2 = Event::out_of_range;

                        // Fill the resolution area.
                        int current_area =
                            data[add_player_suffix(PLAYER_AREA_LABEL,
                                                   player_number + 1)]
                                .at(0, d, t);
                        for (int i = time_event2; i < t; i++) {
                            next_area_per_player2[player_number](d, i) =
                                current_area;
                        }

                        if (marker_event2 == 1) {
                            next_area_per_player2_m1[player_number](d, t) =
                                current_area;
                        }
                        else {
                            next_area_per_player2_m2[player_number](d, t) =
                                current_area;
                        }
                    }

                    if (event2 == Event::within_range) {
                        marker_range_player2[player_number](d, t) = 1;
                    }
                }
            }
        }

        EvidenceSet new_data = data;
        new_data.add_data(add_player_suffix(NEXT_AREA_AFTER_P2_MARKER_LABEL, 1),
                          next_area_per_player1[0]);
        new_data.add_data(add_player_suffix(NEXT_AREA_AFTER_P3_MARKER_LABEL, 1),
                          next_area_per_player2[0]);
        new_data.add_data(add_player_suffix(NEXT_AREA_AFTER_P1_MARKER_LABEL, 2),
                          next_area_per_player1[1]);
        new_data.add_data(add_player_suffix(NEXT_AREA_AFTER_P3_MARKER_LABEL, 2),
                          next_area_per_player2[1]);
        new_data.add_data(add_player_suffix(NEXT_AREA_AFTER_P1_MARKER_LABEL, 3),
                          next_area_per_player1[2]);
        new_data.add_data(add_player_suffix(NEXT_AREA_AFTER_P2_MARKER_LABEL, 3),
                          next_area_per_player2[2]);

        new_data.add_data(
            add_player_suffix(NEXT_AREA_AFTER_P2_MARKER1_LABEL, 1),
            next_area_per_player1_m1[0]);
        new_data.add_data(
            add_player_suffix(NEXT_AREA_AFTER_P3_MARKER1_LABEL, 1),
            next_area_per_player2_m1[0]);
        new_data.add_data(
            add_player_suffix(NEXT_AREA_AFTER_P1_MARKER1_LABEL, 2),
            next_area_per_player1_m1[1]);
        new_data.add_data(
            add_player_suffix(NEXT_AREA_AFTER_P3_MARKER1_LABEL, 2),
            next_area_per_player2_m1[1]);
        new_data.add_data(
            add_player_suffix(NEXT_AREA_AFTER_P1_MARKER1_LABEL, 3),
            next_area_per_player1_m1[2]);
        new_data.add_data(
            add_player_suffix(NEXT_AREA_AFTER_P2_MARKER1_LABEL, 3),
            next_area_per_player2_m1[2]);

        new_data.add_data(
            add_player_suffix(NEXT_AREA_AFTER_P2_MARKER2_LABEL, 1),
            next_area_per_player1_m2[0]);
        new_data.add_data(
            add_player_suffix(NEXT_AREA_AFTER_P3_MARKER2_LABEL, 1),
            next_area_per_player2_m2[0]);
        new_data.add_data(
            add_player_suffix(NEXT_AREA_AFTER_P1_MARKER2_LABEL, 2),
            next_area_per_player1_m2[1]);
        new_data.add_data(
            add_player_suffix(NEXT_AREA_AFTER_P3_MARKER2_LABEL, 2),
            next_area_per_player2_m2[1]);
        new_data.add_data(
            add_player_suffix(NEXT_AREA_AFTER_P1_MARKER2_LABEL, 3),
            next_area_per_player1_m2[2]);
        new_data.add_data(
            add_player_suffix(NEXT_AREA_AFTER_P2_MARKER2_LABEL, 3),
            next_area_per_player2_m2[2]);

        new_data.add_data(add_player_suffix(P2_MARKER_RANGE_LABEL, 1),
                          marker_range_player1[0]);
        new_data.add_data(add_player_suffix(P3_MARKER_RANGE_LABEL, 1),
                          marker_range_player2[0]);
        new_data.add_data(add_player_suffix(P1_MARKER_RANGE_LABEL, 2),
                          marker_range_player1[1]);
        new_data.add_data(add_player_suffix(P3_MARKER_RANGE_LABEL, 2),
                          marker_range_player2[1]);
        new_data.add_data(add_player_suffix(P1_MARKER_RANGE_LABEL, 3),
                          marker_range_player1[2]);
        new_data.add_data(add_player_suffix(P2_MARKER_RANGE_LABEL, 3),
                          marker_range_player2[2]);

        new_data.save(output_dir);
    }
}

void split_report_per_trial(const string& report_filepath,
                            const string& output_dir) {
    // Trials are processed in order so we can treat the file considering that
    // trial messages won't be scrambled.
    unordered_map<string, vector<string>> predictions_per_trial;

    string previous_trial;
    ifstream file(report_filepath);
    if (file.good()) {
        while (!file.eof()) {
            string message;
            getline(file, message);

            if (message == "")
                continue;

            nlohmann::json json_message = nlohmann::json::parse(message);
            string trial_id = json_message["msg"]["trial_number"];
            json_message["msg"].erase("trial_number");

            if (EXISTS(trial_id, predictions_per_trial)) {
                predictions_per_trial[trial_id].push_back(json_message.dump());
            }
            else {
                predictions_per_trial[trial_id] = {json_message.dump()};
            }
        }

        for (const auto& [trial_id, predictions] : predictions_per_trial) {
            string out_filename =
                report_filepath.substr(report_filepath.rfind('/'));
            boost::replace_all(out_filename, "ALL", trial_id);
            string out_filepath = get_filepath(output_dir, out_filename);
            ofstream out_file(out_filepath);
            if (out_file.is_open()) {
                for (const string& prediction : predictions) {
                    out_file << prediction << "\n";
                }
                out_file.close();
            }
        }
    }
}

int main(int argc, char* argv[]) {
    unsigned int option;
    unsigned int num_periods;
    string input_dir;
    string output_dir;
    string external_filepath;

    po::options_description desc("Allowed options");
    desc.add_options()(
        "help,h",
        "This program creates extra data for the study-2_2021.06 evaluation.")(
        "option",
        po::value<unsigned int>(&option),
        "1 - Create TrialPeriod data.\n"
        "2 - Create PlanningBeforeTrial data.\n"
        "3 - Create data for M7 evaluation based on external source. file.\n"
        "4 - Create extra player areas for M7 models.\n"
        "5 - Create next accessed areas based on the nearby markers "
        "observations.\n"
        "6 - Split report with all the predictions into individual reports per "
        "trial.\n")("input_dir",
                    po::value<string>(&input_dir),
                    "Directory where observations are stored.")(
        "output_dir",
        po::value<string>(&output_dir)->required(),
        "Directory new observations must be saved.")(
        "periods",
        po::value<unsigned int>(&num_periods)->default_value(3),
        "Number of periods if option 1 is chosen")(
        "external_filepath",
        po::value<string>(&external_filepath),
        "Filepath of a .csv file needed if option selected is 3 or report with "
        "merged predictions if option selected is 4.");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    if (vm.count("help")) {
        cout << desc << "\n";
        return 1;
    }

    switch (option) {
    case 1:
        create_trial_period_data(input_dir, output_dir, num_periods);
        break;
    case 2:
        create_planning_before_trial_data(input_dir, output_dir);
        break;
    case 3:
        create_m7_data_from_external_source(
            input_dir, output_dir, external_filepath);
        break;
    case 4:
        create_m7_player_areas(input_dir, output_dir);
        break;
    case 5:
        create_observations_for_next_accessed_area(input_dir, output_dir);
        break;
    case 6:
        split_report_per_trial(external_filepath, output_dir);
        break;
    }
}
