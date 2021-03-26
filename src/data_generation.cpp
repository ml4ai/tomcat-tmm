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
        for(const auto& exclusion : nlohmann::json::parse(file)) {
            exclusions.insert((string) exclusion);
        }
    }

    return exclusions;
}

void generate_data(const string& model_json,
                   const string& params_dir,
                   const string& data_dir,
                   const string& exclusions_json,
                   int num_time_steps,
                   int equal_samples_time_step_limit,
                   int num_data_samples,
                   int num_jobs) {

    shared_ptr<gsl_rng> random_generator(gsl_rng_alloc(gsl_rng_mt19937));
    shared_ptr<DynamicBayesNet> model = make_shared<DynamicBayesNet>(
        DynamicBayesNet ::create_from_json(model_json));
    unordered_set<string> exclusions = get_exclusions(exclusions_json);

    Experimentation experimentation(random_generator, "", model);
    experimentation.generate_synthetic_data(params_dir,
                                            data_dir,
                                            num_data_samples,
                                            num_time_steps,
                                            equal_samples_time_step_limit,
                                            exclusions,
                                            num_jobs);
}

int main(int argc, char* argv[]) {
    string model_json;
    string params_dir;
    string data_dir;
    string exclusions_json;
    int num_data_samples;
    int num_time_steps;
    int equal_samples_time_step_limit;
    int num_jobs;

    po::options_description desc("Allowed options");
    desc.add_options()(
        "help,h",
        "This program generates synthetic data from a pre-trained model.")(
        "model_json",
        po::value<string>(&model_json)->required(),
        "Filepath of the json file containing the model definition.")(
        "params_dir",
        po::value<string>(&params_dir)->required(),
        "Directory where the pre-trained model's parameters are saved.")(
        "data_dir",
        po::value<string>(&data_dir)->required(),
        "Directory where the generated data samples must be saved.")(
        "exclusions_json",
        po::value<string>(&exclusions_json),
        "Filepath of a json file containing a list of variable labels for "
        "which data samples must not be saved.")(
        "samples",
        po::value<int>(&num_data_samples)->required(),
        "Number of data samples to generate for each variable (excluding the "
        "ones indicated in the exclusions_json, if any.")(
        "T",
        po::value<int>(&num_time_steps)->default_value(600)->required(),
        "Number of time steps to unroll the model into.")(
        "H",
        po::value<int>(&equal_samples_time_step_limit)->default_value(-1),
        "For each variable in the model, samples up to the time step indicated "
        "in this option will have the same values.")(
        "jobs",
        po::value<int>(&num_jobs)->default_value(4),
        "Number of jobs used for multi-thread training.");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    if (vm.count("help")) {
        cout << desc << "\n";
        return 1;
    }

    generate_data(model_json,
                  params_dir,
                  data_dir,
                  exclusions_json,
                  num_time_steps,
                  equal_samples_time_step_limit,
                  num_data_samples,
                  num_jobs);
}
