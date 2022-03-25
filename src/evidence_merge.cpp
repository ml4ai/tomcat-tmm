#include <fstream>
#include <utility>

#include "pgm/EvidenceSet.h"
#include <boost/program_options.hpp>

using namespace tomcat::model;
using namespace std;
namespace po = boost::program_options;

void merge_sets(const string& set1_dir,
                const string& set2_dir,
                const string& output_dir) {
    EvidenceSet set1(set1_dir);
    EvidenceSet set2(set2_dir);

    set1.merge(set2);
    set1.save(output_dir);
}

int main(int argc, char* argv[]) {
    string set1_dir;
    string set2_dir;
    string output_dir;

    po::options_description desc("Allowed options");
    desc.add_options()("help,h",
                       "This program creates a new evidence set formed by "
                       "merging two evidence sets together.")(
        "set1_dir",
        po::value<string>(&set1_dir)->required(),
        "Directory where the first evidence set is stored.")(
        "set2_dir",
        po::value<string>(&set2_dir)->required(),
        "Directory where the second evidence set is stored.")(
        "output_dir",
        po::value<string>(&output_dir)->required(),
        "Directory where the merged set must be saved.");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    if (vm.count("help")) {
        cout << desc << "\n";
        return 1;
    }

    merge_sets(set1_dir, set2_dir, output_dir);
}
