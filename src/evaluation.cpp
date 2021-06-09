#include <memory>
#include <string>
#include <vector>

#include <boost/program_options.hpp>
#include <gsl/gsl_rng.h>

#include "experiments/Experimentation.h"
#include "pgm/DynamicBayesNet.h"
#include "pgm/EvidenceSet.h"

using namespace tomcat::model;
using namespace std;
namespace po = boost::program_options;

void evaluate(const string& experiment_id,
              const string& model_json,
              const string& params_dir,
              const string& train_dir,
              const string& data_dir,
              const string& eval_dir,
              const string& inference_json,
              int num_folds,
              int num_time_steps,
              int burn_in,
              int num_samples,
              int num_jobs,
              bool baseline,
              bool only_estimates,
              bool exact_inference) {

    shared_ptr<gsl_rng> random_generator(gsl_rng_alloc(gsl_rng_mt19937));
    EvidenceSet test_data(data_dir);
    shared_ptr<DynamicBayesNet> model = make_shared<DynamicBayesNet>(
        DynamicBayesNet ::create_from_json(model_json));
    model->unroll(3, true);

    Experimentation experimentation(random_generator, experiment_id, model);
    experimentation.add_estimators_from_json(inference_json,
                                             burn_in,
                                             num_samples,
                                             num_jobs,
                                             baseline,
                                             exact_inference,
                                             num_time_steps - 1);

    experimentation.evaluate_and_save(params_dir,
                                      num_folds,
                                      eval_dir,
                                      test_data,
                                      baseline,
                                      train_dir,
                                      only_estimates);
}

int main(int argc, char* argv[]) {
    string experiment_id;
    string model_json;
    string params_dir;
    string train_dir;
    string data_dir;
    string eval_dir;
    string inference_json;
    int num_time_steps;
    int num_folds;
    int burn_in;
    int num_samples;
    int num_jobs;
    bool baseline;
    bool only_estimates;
    bool exact_inference;

    po::options_description desc("Allowed options");
    desc.add_options()(
        "help,h",
        "This program evaluates a series of predictions or "
        "inferences over a set of variables and horizons on a pre-trained "
        "model. If the model does not support exact inference (e.g. the model "
        "has at least one variable under a semi-Markov assumption or that "
        "follows a continuous distribution), approximate inference will be "
        "used to estimate the probabilities.")(
        "exp_id", po::value<string>(&experiment_id), "Experiment identifier.")(
        "model_json",
        po::value<string>(&model_json)->required(),
        "Filepath of the json file containing the model definition.")(
        "params_dir",
        po::value<string>(&params_dir)->required(),
        "Directory where the pre-trained model's parameters are saved.")(
        "data_dir",
        po::value<string>(&data_dir)->required(),
        "Directory where the data (evidence) is located.")(
        "eval_dir",
        po::value<string>(&eval_dir)->required(),
        "Directory where the evaluation file must be saved.")(
        "inference_json",
        po::value<string>(&inference_json)->required(),
        "Filepath of the json file containing the variables and inference "
        "horizons to be evaluated by the pre-trained model.")(
        "K",
        po::value<int>(&num_folds)->default_value(1)->required(),
        "Number of folds for evaluation using cross-validation. This assumes "
        "the model was pre-trained using the same number of folds here "
        "defined.")(
        "T",
        po::value<int>(&num_time_steps)->default_value(600)->required(),
        "Number of time steps to unroll the model into.")(
        "burn_in",
        po::value<int>(&burn_in)->default_value(100)->required(),
        "Number of samples to generate until posterior convergence if "
        "approximate inference is used for evaluation.")(
        "samples",
        po::value<int>(&num_samples)->default_value(100)->required(),
        "Number of samples used to estimate the parameters of the model "
        "after the burn-in period if approximate inference is used for "
        "evaluation.")("jobs",
                       po::value<int>(&num_jobs)->default_value(4),
                       "Number of jobs used for multi-thread inference.")(
        "baseline",
        po::bool_switch(&baseline)->default_value(false),
        "Whether the baseline estimator based on frequencies of the samples in"
        " the training data must be used.")(
        "train_dir",
        po::value<string>(&train_dir),
        "Directory where data used for training is. This is only required for"
        " the baseline evaluation.")(
        "only_estimates",
        po::bool_switch(&only_estimates)->default_value(false),
        "If active, performance is not computed. Only the probability "
        "estimates over time.")(
        "exact",
        po::bool_switch(&exact_inference)->default_value(false),
        "Whether to use exact or approximate inference.");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    if (vm.count("help")) {
        cout << desc << endl;
        return 1;
    }
    else if (baseline && train_dir == "" && num_folds == 1) {
        cout << "For baseline evaluation without cross validation, the "
                "directory of the data used for training the model must be "
                "informed."
             << endl;
        return 1;
    }

    evaluate(experiment_id,
             model_json,
             params_dir,
             train_dir,
             data_dir,
             eval_dir,
             inference_json,
             num_folds,
             num_time_steps,
             burn_in,
             num_samples,
             num_jobs,
             baseline,
             only_estimates,
             exact_inference);
}
