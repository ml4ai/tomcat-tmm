#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include <boost/program_options.hpp>
#include <gsl/gsl_rng.h>
#include <nlohmann/json.hpp>

#include "experiments/Experimentation.h"
#include "pgm/DynamicBayesNet.h"
#include "pgm/EvidenceSet.h"

using namespace tomcat::model;
using namespace std;
namespace po = boost::program_options;

static unordered_set<string> get_exclusions(const string& filepath) {
    unordered_set<string> exclusions;

    fstream file;
    file.open(filepath);
    if (file.is_open()) {
        for (const auto& exclusion : nlohmann::json::parse(file)) {
            exclusions.insert((string)exclusion);
        }
    }

    return exclusions;
}

void print_model(const string& model_json,
                 const string& params_dir,
                 const string& model_dir) {

    shared_ptr<DynamicBayesNet> model = make_shared<DynamicBayesNet>(
        DynamicBayesNet ::create_from_json(model_json));

    Experimentation experimentation({}, "", model);
    experimentation.print_model(params_dir, model_dir);
}

int main(int argc, char* argv[]) {
    string model_json;
    string params_dir;
    string model_dir;

    po::options_description desc("Allowed options");
    desc.add_options()("help,h",
                       "This program prints the structure of a model.")(
        "model_json",
        po::value<string>(&model_json)->required(),
        "Filepath of the json file containing the model definition.")(
        "params_dir",
        po::value<string>(&params_dir),
        "Directory where the pre-trained model's parameters are saved. This "
        "is only required if the CPDs are to be printed as well.")(
        "model_dir",
        po::value<string>(&model_dir),
        "Directory where the files containing the model's structure and/or "
        "CPDs must be saved.");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    if (vm.count("help")) {
        cout << desc << "\n";
        return 1;
    }

    print_model(model_json, params_dir, model_dir);
}
