#include <memory>
#include <string>
#include <vector>
#include <set>
#include <utility>

#include <boost/program_options.hpp>

#include "converter/ASISTMultiPlayerMessageConverter.h"
#include "pgm/EvidenceSet.h"

using namespace tomcat::model;
using namespace std;
namespace po = boost::program_options;

void convert(const string& input_dir, const string& output_dir) {
    EvidenceSet time_data(input_dir);

    // Remove variables that are not relevant for the event based model
    time_data.remove(ASISTMultiPlayerMessageConverter::FINAL_TEAM_SCORE_LABEL);
    time_data.remove(
        ASISTMultiPlayerMessageConverter::MAP_VERSION_ASSIGNMENT_LABEL);
    time_data.remove(ASISTMultiPlayerMessageConverter::TEAM_SCORE_LABEL);
    for (int player_num = 1; player_num <= 3; player_num++) {
        time_data.remove(
            ASISTMultiPlayerMessageConverter::get_player_variable_label(
                ASISTMultiPlayerMessageConverter::OBS_PLAYER_BUILDING_SECTION_LABEL,
                player_num));
        time_data.remove(
            ASISTMultiPlayerMessageConverter::get_player_variable_label(
                ASISTMultiPlayerMessageConverter::OBS_PLAYER_EXPANDED_BUILDING_SECTION_LABEL,
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

    vector<EvidenceSet> event_data_per_trial(time_data.get_num_data_points());
    enum Event { within_range, out_of_range };
    int max_events = 0;
    vector<vector<pair<int, int>>> time_2_event_per_trial(time_data.get_num_data_points());
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
                time_2_event_per_trial[d].push_back({t, event_data_per_trial[d].get_time_steps()});
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

    event_data.save(output_dir);
}

int main(int argc, char* argv[]) {
    string input_dir;
    string output_dir;

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
        "Directory where the event-based data must be saved.");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    if (vm.count("help")) {
        cout << desc << "\n";
        return 1;
    }

    convert(input_dir, output_dir);
}
