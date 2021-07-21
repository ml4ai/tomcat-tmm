#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <boost/program_options.hpp>

#include "converter/ASISTMultiPlayerMessageConverter.h"
#include "pgm/EvidenceSet.h"

using namespace tomcat::model;
using namespace std;
namespace po = boost::program_options;
#define add_player_suffix(label, player_num)                                   \
    ASISTMultiPlayerMessageConverter::get_player_variable_label(label,         \
                                                                player_num)

EvidenceSet convert_including_players(const EvidenceSet& time_data) {
    vector<EvidenceSet> event_data_per_trial(time_data.get_num_data_points());
    enum Event { within_range, out_of_range };
    int max_events = 0;
    vector<vector<pair<int, int>>> time_2_event_per_trial(
        time_data.get_num_data_points());
    for (int d = 0; d < time_data.get_num_data_points(); d++) {
        vector<Event> event = {out_of_range, out_of_range, out_of_range};
        EvidenceSet single_point_time_data = time_data.get_single_point_data(d);

        for (int t = 0; t < time_data.get_time_steps(); t++) {
            bool event_detected = false;
            for (int player_num = 1; player_num <= 3; player_num++) {
                int observed_marker =
                    single_point_time_data
                        [ASISTMultiPlayerMessageConverter::
                             get_player_variable_label(
                                 ASISTMultiPlayerMessageConverter::
                                     OTHER_PLAYER_NEARBY_MARKER,
                                 player_num)]
                            .at(0, 0, t);
                if ((event[player_num - 1] == out_of_range &&
                     observed_marker > 0) ||
                    (event[player_num - 1] == within_range &&
                     observed_marker == 0)) {

                    if (event[player_num - 1] == out_of_range) {
                        event[player_num - 1] = within_range;
                    }
                    else {
                        event[player_num - 1] = out_of_range;
                    }

                    event_detected = true;
                }
            }

            if (event_detected) {
                time_2_event_per_trial[d].push_back(
                    {t, event_data_per_trial[d].get_time_steps()});
                event_data_per_trial[d].hstack(
                    single_point_time_data.get_single_time_data(t));
            }
        }

        max_events = max(max_events, event_data_per_trial[d].get_time_steps());
    }

    // Complete trials with dummy values so that all trials have the same number
    // of columns (events).
    EvidenceSet event_data(true);
    event_data.set_time_2_event_per_data_point(time_2_event_per_trial);

    for (EvidenceSet event_trial : event_data_per_trial) {
        int events_to_add = max_events - event_trial.get_time_steps();

        if (events_to_add > 0) {
            EvidenceSet complement;

            complement.add_data(ASISTMultiPlayerMessageConverter::
                                    MARKER_LEGEND_ASSIGNMENT_LABEL,
                                Tensor3::constant(1, 1, events_to_add, 3));

            complement.add_data(
                ASISTMultiPlayerMessageConverter::PLANNING_CONDITION_LABEL,
                Tensor3::constant(1, 1, events_to_add, 2));

            for (int player_num = 1; player_num <= 3; player_num++) {

                complement.add_data(
                    ASISTMultiPlayerMessageConverter::get_player_variable_label(
                        ASISTMultiPlayerMessageConverter::PLAYER_ROLE_LABEL,
                        player_num),
                    Tensor3::constant(1, 1, events_to_add, 3));

                complement.add_data(
                    ASISTMultiPlayerMessageConverter::get_player_variable_label(
                        ASISTMultiPlayerMessageConverter::
                            OTHER_PLAYER_NEARBY_MARKER,
                        player_num),
                    Tensor3::constant(1, 1, events_to_add, 4));

                complement.add_data(
                    ASISTMultiPlayerMessageConverter::get_player_variable_label(
                        ASISTMultiPlayerMessageConverter::PLAYER_AREA_LABEL,
                        player_num),
                    Tensor3::constant(1, 1, events_to_add, 2));

                complement.add_data(
                    ASISTMultiPlayerMessageConverter::get_player_variable_label(
                        ASISTMultiPlayerMessageConverter::
                            PLAYER_MARKER_LEGEND_VERSION_LABEL,
                        player_num),
                    Tensor3::constant(1, 1, events_to_add, 2));
            }

            event_trial.hstack(complement);
        }

        event_data.vstack(event_trial);
    }

    return event_data;
}

EvidenceSet convert_merging_players(const EvidenceSet& time_data, unsigned int player_num) {
    vector<EvidenceSet> event_data_per_trial;
    vector<vector<pair<int, int>>> time_2_event_per_trial;
    enum Event { within_range, out_of_range };
    int max_events = 0;

    vector<int> player_nums;
    if (player_num == 0) {
        player_nums = {1, 2, 3};
    } else {
        player_nums.push_back(player_num);
    }

    for (int d = 0; d < time_data.get_num_data_points(); d++) {
        EvidenceSet single_point_time_data = time_data.get_single_point_data(d);

        for (int player_num : player_nums) {
            EvidenceSet event_data_per_player(true);
            Event event = out_of_range;
            vector<pair<int, int>> time_2_event_per_player;

            for (int t = 0; t < time_data.get_time_steps(); t++) {
                int observed_marker =
                    single_point_time_data
                        [ASISTMultiPlayerMessageConverter::
                             get_player_variable_label(
                                 ASISTMultiPlayerMessageConverter::
                                     OTHER_PLAYER_NEARBY_MARKER,
                                 player_num)]
                            .at(0, 0, t);
                if ((event == out_of_range && observed_marker > 0) ||
                    (event == within_range && observed_marker == 0)) {

                    if (event == out_of_range) {
                        event = within_range;
                    }
                    else {
                        event = out_of_range;
                    }

                    EvidenceSet single_time_data =
                        single_point_time_data.get_single_time_data(t);
                    EvidenceSet single_event_data(true);

                    string marker_legend_label =
                        ASISTMultiPlayerMessageConverter::
                            MARKER_LEGEND_ASSIGNMENT_LABEL;
                    string planning_condition_label =
                        ASISTMultiPlayerMessageConverter::
                            PLANNING_CONDITION_LABEL;

                    single_event_data.add_data(
                        marker_legend_label,
                        single_time_data[marker_legend_label]);
                    single_event_data.add_data(
                        planning_condition_label,
                        single_time_data[planning_condition_label]);

                    string player_role_label = add_player_suffix(
                        ASISTMultiPlayerMessageConverter::PLAYER_ROLE_LABEL,
                        player_num);
                    string player_nearby_marker_label =
                        add_player_suffix(ASISTMultiPlayerMessageConverter::
                                              OTHER_PLAYER_NEARBY_MARKER,
                                          player_num);
                    string player_area_label = add_player_suffix(
                        ASISTMultiPlayerMessageConverter::PLAYER_AREA_LABEL,
                        player_num);
                    string player_marker_legend_label = add_player_suffix(
                        ASISTMultiPlayerMessageConverter::
                            PLAYER_MARKER_LEGEND_VERSION_LABEL,
                        player_num);

                    // The player variables do not make a distinction between
                    // players in this setting. Each player will be added as a
                    // different data point in the final set.
                    single_event_data.add_data(
                        ASISTMultiPlayerMessageConverter::PLAYER_ROLE_LABEL,
                        single_time_data[player_role_label]);
                    single_event_data.add_data(
                        ASISTMultiPlayerMessageConverter::
                            OTHER_PLAYER_NEARBY_MARKER,
                        single_time_data[player_nearby_marker_label]);
                    single_event_data.add_data(
                        ASISTMultiPlayerMessageConverter::PLAYER_AREA_LABEL,
                        single_time_data[player_area_label]);
                    single_event_data.add_data(
                        ASISTMultiPlayerMessageConverter::
                            PLAYER_MARKER_LEGEND_VERSION_LABEL,
                        single_time_data[player_marker_legend_label]);

                    time_2_event_per_player.push_back(
                        {t, event_data_per_player.get_time_steps()});
                    event_data_per_player.hstack(single_event_data);
                }
            }

            event_data_per_trial.push_back(event_data_per_player);
            time_2_event_per_trial.push_back(time_2_event_per_player);
            max_events =
                max(max_events, event_data_per_player.get_time_steps());
        }
    }

    // Complete trials with dummy values so that all trials have the same number
    // of columns (events).
    EvidenceSet event_data(true);
    event_data.set_time_2_event_per_data_point(time_2_event_per_trial);

    for (EvidenceSet event_trial : event_data_per_trial) {
        int events_to_add = max_events - event_trial.get_time_steps();

        if (events_to_add > 0) {
            EvidenceSet complement;

            complement.add_data(ASISTMultiPlayerMessageConverter::
                                    MARKER_LEGEND_ASSIGNMENT_LABEL,
                                Tensor3::constant(1, 1, events_to_add, 3));

            complement.add_data(
                ASISTMultiPlayerMessageConverter::PLANNING_CONDITION_LABEL,
                Tensor3::constant(1, 1, events_to_add, 2));

            complement.add_data(
                ASISTMultiPlayerMessageConverter::PLAYER_ROLE_LABEL,
                Tensor3::constant(1, 1, events_to_add, 3));

            complement.add_data(
                ASISTMultiPlayerMessageConverter::OTHER_PLAYER_NEARBY_MARKER,
                Tensor3::constant(1, 1, events_to_add, 4));

            complement.add_data(
                ASISTMultiPlayerMessageConverter::PLAYER_AREA_LABEL,
                Tensor3::constant(1, 1, events_to_add, 2));

            complement.add_data(ASISTMultiPlayerMessageConverter::
                                    PLAYER_MARKER_LEGEND_VERSION_LABEL,
                                Tensor3::constant(1, 1, events_to_add, 2));

            event_trial.hstack(complement);
        }

        event_data.vstack(event_trial);
    }

    return event_data;
}

void convert(const string& input_dir,
             const string& output_dir,
             unsigned int player_option,
             unsigned int player_num) {
    EvidenceSet time_data(input_dir);

    // Remove variables that are not relevant for the event based model
    time_data.remove(ASISTMultiPlayerMessageConverter::FINAL_TEAM_SCORE_LABEL);
    time_data.remove(
        ASISTMultiPlayerMessageConverter::MAP_VERSION_ASSIGNMENT_LABEL);
    time_data.remove(ASISTMultiPlayerMessageConverter::TEAM_SCORE_LABEL);
    for (int player_num = 1; player_num <= 3; player_num++) {
        time_data.remove(
            ASISTMultiPlayerMessageConverter::get_player_variable_label(
                ASISTMultiPlayerMessageConverter::
                    OBS_PLAYER_BUILDING_SECTION_LABEL,
                player_num));
        time_data.remove(
            ASISTMultiPlayerMessageConverter::get_player_variable_label(
                ASISTMultiPlayerMessageConverter::
                    OBS_PLAYER_EXPANDED_BUILDING_SECTION_LABEL,
                player_num));
        time_data.remove(
            ASISTMultiPlayerMessageConverter::get_player_variable_label(
                ASISTMultiPlayerMessageConverter::PLAYER_MAP_VERSION_LABEL,
                player_num));
        time_data.remove(
            ASISTMultiPlayerMessageConverter::get_player_variable_label(
                ASISTMultiPlayerMessageConverter::PLAYER_PLACED_MARKER_LABEL,
                player_num));
        time_data.remove(
            ASISTMultiPlayerMessageConverter::get_player_variable_label(
                ASISTMultiPlayerMessageConverter::PLAYER_TASK_LABEL,
                player_num));
    }

    EvidenceSet event_data;

    if (player_option == 0) {
        event_data = convert_including_players(time_data);
    }
    else if (player_option == 1) {
        event_data = convert_merging_players(time_data, 0);
    } else {
        event_data = convert_merging_players(time_data, player_num);
    }
    event_data.save(output_dir);
}

int main(int argc, char* argv[]) {
    string input_dir;
    string output_dir;
    unsigned int player_option;
    unsigned int player_num;

    po::options_description desc("Allowed options");
    desc.add_options()(
        "help,h",
        "This program converts time-step-based data to event-based data "
        "according to markers in players vicinity.")(
        "input_dir",
        po::value<string>(&input_dir)->required(),
        "Directory where the time-step--based data is located.")(
        "output_dir",
        po::value<string>(&output_dir)->required(),
        "Directory where the event-based data must be saved.")(
        "player_option",
        po::value<unsigned int>(&player_option),
        "0 - Include players \n"
        "1 - Merge players \n"
        "2 - Include a specific player")(
        "player_num",
        po::value<unsigned int>(&player_num),
        "Player to include if option 2 was selected");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    if (vm.count("help")) {
        cout << desc << "\n";
        return 1;
    }

    convert(input_dir, output_dir, player_option, player_num);
}
