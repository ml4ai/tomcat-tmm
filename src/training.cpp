#include <string>
#include <vector>
#include <memory>

#include <boost/program_options.hpp>
#include <gsl/gsl_rng.h>

#include "pgm/DynamicBayesNet.h"
#include "experiments/Experimentation.h"
#include "pgm/EvidenceSet.h"

using namespace tomcat::model;
using namespace std;
namespace po = boost::program_options;

void train(const string& model_json,
           const string& data_dir,
           const string& params_dir,
           int num_folds,
           int num_time_steps,
           int burn_in,
           int num_samples,
           int num_jobs) {

    shared_ptr<gsl_rng> random_generator(gsl_rng_alloc(gsl_rng_mt19937));
    EvidenceSet training_data(data_dir);
    shared_ptr<DynamicBayesNet> model = make_shared<DynamicBayesNet>(
        DynamicBayesNet ::create_from_json(model_json));
    model->unroll(num_time_steps, true);

//    training_data.keep_first(1);

    Experimentation experimentation(random_generator, "", model);
    experimentation.set_gibbs_trainer(burn_in, num_samples, num_jobs);
    experimentation.train_and_save(params_dir, num_folds, training_data);
}

int main(int argc, char* argv[]) {
    string model_json;
    string data_dir;
    string params_dir;
    int num_folds;
    int num_time_steps;
    int burn_in;
    int num_samples;
    int num_jobs;

    po::options_description desc("Allowed options");
    desc.add_options()(
        "help,h",
        "This program trains a model via Gibbs sampling procedure, and "
        "saves the learned parameters to an informed directory.")(
        "model_json",
        po::value<string>(&model_json)->required(),
        "Filepath of the json file containing the model definition.")(
        "data_dir",
        po::value<string>(&data_dir)->required(),
        "Directory where the data (evidence) is located.")(
        "params_dir",
        po::value<string>(&params_dir)->required(),
        "Directory where the trained model's parameters must be saved.")(
        "K",
        po::value<int>(&num_folds)->default_value(1),
        "Number of folds for cross-validation training. If set to 1, the full"
        " training data is used.")(
        "T",
        po::value<int>(&num_time_steps)->default_value(600)->required(),
        "Number of time steps to unroll the model into.")(
        "burn_in",
        po::value<int>(&burn_in)->default_value(100)->required(),
        "Number of samples to generate until posterior convergence.")(
        "samples",
        po::value<int>(&num_samples)->default_value(100)->required(),
        "Number of samples used to estimate the parameters of the model "
        "after the burn-in period.")(
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

    train(model_json,
          data_dir,
          params_dir,
          num_folds,
          num_time_steps,
          burn_in,
          num_samples,
          num_jobs);
}
