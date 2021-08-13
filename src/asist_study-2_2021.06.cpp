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
#define P1_NEARBY_MARKER1_LABEL "Player1NearbyMarker1"
#define P2_NEARBY_MARKER1_LABEL "Player2NearbyMarker1"
#define P3_NEARBY_MARKER1_LABEL "Player3NearbyMarker1"
#define P1_NEARBY_MARKER2_LABEL "Player1NearbyMarker2"
#define P2_NEARBY_MARKER2_LABEL "Player2NearbyMarker2"
#define P3_NEARBY_MARKER2_LABEL "Player3NearbyMarker2"

const inline string PLAYER1_PLAYER_MARKER_AREA_LABEL =
    "Player1PlayerMarkerArea";
const inline string PLAYER2_PLAYER_MARKER_AREA_LABEL =
    "Player2PlayerMarkerArea";
const inline string PLAYER3_PLAYER_MARKER_AREA_LABEL =
    "Player3PlayerMarkerArea";

enum Event { within_range, out_of_range };

// Auxiliary function
void fill_next_area(Eigen::MatrixXd& next_areas,
                    int data_point,
                    int time_step,
                    int other_player_idx,
                    int marker,
                    int area,
                    vector<int>& event,
                    vector<int>& event_start) {
    if (event[other_player_idx] == Event::out_of_range && marker != NO_MARKER) {
        event[other_player_idx] = Event::within_range;
        event_start[other_player_idx] = time_step;

        // Temporary. This will be overridden when the player leaves the
        // marker's proximity.
        next_areas(data_point, time_step) = area;
    }
    else if (event[other_player_idx] == Event::within_range &&
             marker == NO_MARKER) {
        int start_time = event_start[other_player_idx];
        next_areas(data_point, start_time) = area;
        event[other_player_idx] = Event::out_of_range;
    }
}

void create_next_area_on_nearby_marker_data(const string& input_dir,
                                            const string& output_dir) {
    EvidenceSet data(input_dir);

    // Markers placed by others label
    string player_2_nearby_marker_p1_label = add_player_suffix(
        ASISTMultiPlayerMessageConverter::PLAYER2_NEARBY_MARKER_LABEL, 1);
    string player_3_nearby_marker_p1_label = add_player_suffix(
        ASISTMultiPlayerMessageConverter::PLAYER3_NEARBY_MARKER_LABEL, 1);
    string player_1_nearby_marker_p2_label = add_player_suffix(
        ASISTMultiPlayerMessageConverter::PLAYER1_NEARBY_MARKER_LABEL, 2);
    string player_3_nearby_marker_p2_label = add_player_suffix(
        ASISTMultiPlayerMessageConverter::PLAYER3_NEARBY_MARKER_LABEL, 2);
    string player_1_nearby_marker_p3_label = add_player_suffix(
        ASISTMultiPlayerMessageConverter::PLAYER1_NEARBY_MARKER_LABEL, 3);
    string player_2_nearby_marker_p3_label = add_player_suffix(
        ASISTMultiPlayerMessageConverter::PLAYER2_NEARBY_MARKER_LABEL, 3);

    // Areas accessed by the players label
    string area_p1_label = add_player_suffix(
        ASISTMultiPlayerMessageConverter::PLAYER_AREA_LABEL, 1);
    string area_p2_label = add_player_suffix(
        ASISTMultiPlayerMessageConverter::PLAYER_AREA_LABEL, 2);
    string area_p3_label = add_player_suffix(
        ASISTMultiPlayerMessageConverter::PLAYER_AREA_LABEL, 3);

    Eigen::MatrixXd next_area_after_p2_marker_p1 = Eigen::MatrixXd::Constant(
        data.get_num_data_points(), data.get_time_steps(), NO_OBS);
    Eigen::MatrixXd next_area_after_p3_marker_p1 = Eigen::MatrixXd::Constant(
        data.get_num_data_points(), data.get_time_steps(), NO_OBS);
    Eigen::MatrixXd next_area_after_p1_marker_p2 = Eigen::MatrixXd::Constant(
        data.get_num_data_points(), data.get_time_steps(), NO_OBS);
    Eigen::MatrixXd next_area_after_p3_marker_p2 = Eigen::MatrixXd::Constant(
        data.get_num_data_points(), data.get_time_steps(), NO_OBS);
    Eigen::MatrixXd next_area_after_p1_marker_p3 = Eigen::MatrixXd::Constant(
        data.get_num_data_points(), data.get_time_steps(), NO_OBS);
    Eigen::MatrixXd next_area_after_p2_marker_p3 = Eigen::MatrixXd::Constant(
        data.get_num_data_points(), data.get_time_steps(), NO_OBS);

    // Whether the player is withing the range of detection of markers places by
    // the two other players.
    vector<int> player_1_marker_event(2, Event::out_of_range);
    vector<int> player_2_marker_event(2, Event::out_of_range);
    vector<int> player_3_marker_event(2, Event::out_of_range);

    // Time when an an active event is triggered
    vector<int> player_1_marker_event_start(2, 0);
    vector<int> player_2_marker_event_start(2, 0);
    vector<int> player_3_marker_event_start(2, 0);

    for (int d = 0; d < data.get_num_data_points(); d++) {
        for (int t = 0; t < data.get_time_steps(); t++) {
            // Player 1
            int area_p1 = data[area_p1_label].at(0, d, t);
            int marker_placed_by_p2 =
                data[player_2_nearby_marker_p1_label].at(0, d, t);
            fill_next_area(next_area_after_p2_marker_p1,
                           d,
                           t,
                           0,
                           marker_placed_by_p2,
                           area_p1,
                           player_1_marker_event,
                           player_1_marker_event_start);

            int marker_placed_by_p3 =
                data[player_3_nearby_marker_p1_label].at(0, d, t);
            fill_next_area(next_area_after_p3_marker_p1,
                           d,
                           t,
                           1,
                           marker_placed_by_p3,
                           area_p1,
                           player_1_marker_event,
                           player_1_marker_event_start);

            // Player 2
            int area_p2 = data[area_p2_label].at(0, d, t);
            int marker_placed_by_p1 =
                data[player_1_nearby_marker_p2_label].at(0, d, t);
            fill_next_area(next_area_after_p1_marker_p2,
                           d,
                           t,
                           0,
                           marker_placed_by_p1,
                           area_p2,
                           player_2_marker_event,
                           player_2_marker_event_start);

            marker_placed_by_p3 =
                data[player_3_nearby_marker_p2_label].at(0, d, t);
            fill_next_area(next_area_after_p3_marker_p2,
                           d,
                           t,
                           1,
                           marker_placed_by_p3,
                           area_p2,
                           player_2_marker_event,
                           player_2_marker_event_start);

            // Player 3
            int area_p3 = data[area_p3_label].at(0, d, t);
            marker_placed_by_p1 =
                data[player_1_nearby_marker_p3_label].at(0, d, t);
            fill_next_area(next_area_after_p1_marker_p3,
                           d,
                           t,
                           0,
                           marker_placed_by_p1,
                           area_p3,
                           player_3_marker_event,
                           player_3_marker_event_start);

            marker_placed_by_p2 =
                data[player_2_nearby_marker_p3_label].at(0, d, t);
            fill_next_area(next_area_after_p2_marker_p3,
                           d,
                           t,
                           1,
                           marker_placed_by_p2,
                           area_p3,
                           player_3_marker_event,
                           player_3_marker_event_start);
        }
    }

    string player_2_nearby_marker_p1_next_area_label =
        add_player_suffix("NextAreaAfterP2Marker", 1);
    string player_3_nearby_marker_p1_next_area_label =
        add_player_suffix("NextAreaAfterP3Marker", 1);
    string player_1_nearby_marker_p2_next_area_label =
        add_player_suffix("NextAreaAfterP1Marker", 2);
    string player_3_nearby_marker_p2_next_area_label =
        add_player_suffix("NextAreaAfterP3Marker", 2);
    string player_1_nearby_marker_p3_next_area_label =
        add_player_suffix("NextAreaAfterP1Marker", 3);
    string player_2_nearby_marker_p3_next_area_label =
        add_player_suffix("NextAreaAfterP2Marker", 3);

    EvidenceSet next_areas;
    next_areas.add_data(player_2_nearby_marker_p1_next_area_label,
                        next_area_after_p2_marker_p1);
    next_areas.add_data(player_3_nearby_marker_p1_next_area_label,
                        next_area_after_p3_marker_p1);
    next_areas.add_data(player_1_nearby_marker_p2_next_area_label,
                        next_area_after_p1_marker_p2);
    next_areas.add_data(player_3_nearby_marker_p2_next_area_label,
                        next_area_after_p3_marker_p2);
    next_areas.add_data(player_1_nearby_marker_p3_next_area_label,
                        next_area_after_p1_marker_p3);
    next_areas.add_data(player_2_nearby_marker_p3_next_area_label,
                        next_area_after_p2_marker_p3);

    next_areas.save(output_dir);
}

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

        vector<Eigen::MatrixXd> area_per_player(3);

        vector<Eigen::MatrixXd> marker_area_per_player1(
            3, Eigen::MatrixXd::Constant(num_rows, num_cols, NO_OBS));
        vector<Eigen::MatrixXd> marker_area_per_player2(
            3, Eigen::MatrixXd::Constant(num_rows, num_cols, NO_OBS));

        area_per_player[0] =
            data[add_player_suffix(PLAYER_AREA_LABEL, 1)](0, 0);
        area_per_player[1] =
            data[add_player_suffix(PLAYER_AREA_LABEL, 2)](0, 0);
        area_per_player[2] =
            data[add_player_suffix(PLAYER_AREA_LABEL, 3)](0, 0);

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
                int final_time_step =
                    min(resolved_elapsed_milliseconds / 1000, num_cols - 1);
                if (resolved_elapsed_milliseconds < 0) {
                    // Estimate the player is going to be around the marker for
                    // 2 seconds. This is necessary so we can capture the right
                    // player next area in the test set. The event might stop
                    // before the player enters the room.
                    final_time_step = time_step + 2;
                }
                else if (final_time_step == time_step) {
                    final_time_step = time_step + 1;
                }

                final_time_step = min(final_time_step, num_cols);

                // Make the area where the player is consistent with the event.
                // It always happens when the player is in the hallway.
                for (int t = time_step; t < final_time_step; t++) {
                    area_per_player[player_number](d, t) = HALLWAY;
                }

                if (final_time_step < num_cols) {
                    if (m7_event.next_area == NO_OBS) {
                        // Area is not given in the test file. We just try
                        // no make observations from the game more
                        // synchronized with the events given because they
                        // are measured in different time scales. Accurate
                        // observations are important because of the belief
                        // update over time. The model has to use
                        // information about what happened to improve its
                        // future predictions.
                        if (area_per_player[player_number](
                                d, final_time_step - 1) == ROOM) {
                            // The measurements are taken in milliseconds
                            // and the data in seconds. Therefore, if the
                            // player is in a room at the second of a
                            // measurement, this means this is his next
                            // area should be in the room.
                            area_per_player[player_number](d, final_time_step) =
                                ROOM;
                            area_per_player[player_number](
                                d, final_time_step - 1) = HALLWAY;
                        }
                    }
                    else {
                        // Area is given in the training file. We just use
                        // it.
                        area_per_player[player_number](d, final_time_step) =
                            m7_event.next_area;
                    }
                }

                // Include the new event
                if ((player_number == 0 && other_player_number == 1) ||
                    (player_number == 1 && other_player_number == 0) ||
                    (player_number == 2 && other_player_number == 0)) {

                    for (int t = time_step; t < final_time_step; t++) {
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

                    if (final_time_step < num_cols) {
                        marker_area_per_player1[player_number](
                            d, final_time_step) =
                            area_per_player[player_number](d, final_time_step);
                    }
                }
                else {
                    for (int t = time_step; t < final_time_step; t++) {
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

                        if (final_time_step < num_cols) {
                            marker_area_per_player2[player_number](
                                d, final_time_step) =
                                area_per_player[player_number](d,
                                                               final_time_step);
                        }
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

        new_data.add_data(add_player_suffix(PLAYER_AREA_LABEL, 1),
                          area_per_player[0]);
        new_data.add_data(add_player_suffix(PLAYER_AREA_LABEL, 2),
                          area_per_player[1]);
        new_data.add_data(add_player_suffix(PLAYER_AREA_LABEL, 3),
                          area_per_player[2]);

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

        new_data.add_data(
            add_player_suffix(PLAYER2_PLAYER_MARKER_AREA_LABEL, 1),
            marker_area_per_player1[0]);
        new_data.add_data(
            add_player_suffix(PLAYER3_PLAYER_MARKER_AREA_LABEL, 1),
            marker_area_per_player2[0]);
        new_data.add_data(
            add_player_suffix(PLAYER1_PLAYER_MARKER_AREA_LABEL, 2),
            marker_area_per_player1[1]);
        new_data.add_data(
            add_player_suffix(PLAYER3_PLAYER_MARKER_AREA_LABEL, 2),
            marker_area_per_player2[1]);
        new_data.add_data(
            add_player_suffix(PLAYER1_PLAYER_MARKER_AREA_LABEL, 3),
            marker_area_per_player1[2]);
        new_data.add_data(
            add_player_suffix(PLAYER2_PLAYER_MARKER_AREA_LABEL, 3),
            marker_area_per_player2[2]);

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
        "0 - Create observations for the next area after a nearby marker event "
        "detection.\n"
        "1 - Create TrialPeriod data.\n"
        "2 - Create PlanningBeforeTrial data.\n"
        "3 - Create data for M7 evaluation based on external source. file.\n"
        "4 - Split report with all the predictions into individual reports per "
        "trial.\n"
        "5 - Create extra player areas for M7 models.")(
        "input_dir",
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

//    switch (option) {
//    case 0:
//        create_next_area_on_nearby_marker_data(input_dir, output_dir);
//        break;
//    case 1:
//        create_trial_period_data(input_dir, output_dir, num_periods);
//        break;
//    case 2:
//        create_planning_before_trial_data(input_dir, output_dir);
//        break;
//    case 3:
//        create_m7_data_from_external_source(
//            input_dir, output_dir, external_filepath);
//        break;
//    case 4:
//        split_report_per_trial(external_filepath, output_dir);
//        break;
//    case 5:
//        create_m7_player_areas(input_dir, output_dir);
//        break;
//    }

        EvidenceSet evidence(
            "../../data/asist/study-2_2021.06/evidence/val", false);
        evidence.keep_only(4);
        evidence.save("../../data/asist/study-2_2021.06/evidence/val_single");

        EvidenceSet evidence2(
            "../../data/asist/study-2_2021.06/evidence/val", false);
        evidence2.keep_only(4);
        evidence2.save("../../data/asist/study-2_2021.06/evidence/train_single");
}
