/**
 * This program is responsible for converting messages from the message bus to a
 * matricial format interpretable by a probabilistic model.
 */

#include <string>

#include <boost/program_options.hpp>

#include "converter/TA3MessageConverter.h"

using namespace tomcat::model;
using namespace std;
namespace po = boost::program_options;

void extract_data_from_messages(const string& map_json,
                                const string& messages_dir,
                                const string& data_dir,
                                bool multiplayer,
                                int num_seconds,
                                int time_step_size) {

    TA3MessageConverter converter(num_seconds, time_step_size, map_json);
    converter.convert_messages(messages_dir, data_dir);
}

int main(int argc, char* argv[]) {
    string map_json;
    string messages_dir;
    string data_dir;
    bool multiplayer;
    int num_seconds;
    int time_step_size;

    po::options_description desc("Allowed options");
    desc.add_options()("help,h",
                       "This executable converts messages from the "
                       "ASIST test bed to the matrix format that"
                       " can be interpreted by a DBN.")(
        "map_json",
        po::value<string>(&map_json)->required(),
        "Path to the json file containing the map configuration.")(
        "messages_dir",
        po::value<string>(&messages_dir)->required(),
        "Directory where the files with the messages are stored.")(
        "data_dir",
        po::value<string>(&data_dir)->required(),
        "Directory where the data must be saved.")(
        "multiplayer",
        po::bool_switch(&multiplayer)->required(),
        "Whether the messages come from a multiplayer mission.")(
        "seconds",
        po::value<int>(&num_seconds)->required(),
        "Number of seconds in the mission.")(
        "step_size",
        po::value<int>(&time_step_size)->default_value(1)->required(),
        "Size of a time step in seconds.");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    if (vm.count("help")) {
        cout << desc << "\n";
        return 1;
    }

    extract_data_from_messages(map_json,
                               messages_dir,
                               data_dir,
                               multiplayer,
                               num_seconds,
                               time_step_size);
}
