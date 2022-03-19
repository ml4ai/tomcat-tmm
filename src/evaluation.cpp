#include <memory>

#include <boost/program_options.hpp>
#include <gsl/gsl_rng.h>

#include "experiments/Experimentation.h"
#include "pgm/EvidenceSet.h"
#include "reporter/EstimateReporter.h"

using namespace tomcat::model;
using namespace std;
namespace po = boost::program_options;

void evaluate(const string& experiment_id,
              const string& model_dir,
              const string& params_dir,
              const string& train_dir,
              const string& data_dir,
              const string& eval_dir,
              const string& agent_json,
              const string& reporter_type,
              int num_folds,
              int num_time_steps,
              int num_jobs) {

    shared_ptr<gsl_rng> random_generator(gsl_rng_alloc(gsl_rng_mt19937));

    Experimentation experimentation(random_generator, experiment_id);

    EstimateReporterPtr reporter = EstimateReporter::factory(reporter_type);
    if (reporter) {
        //        reporter->set_json_settings(reporter_json_settings);
    }
    experimentation.set_offline_estimation_process(agent_json,
                                                   model_dir,
                                                   eval_dir,
                                                   num_jobs,
                                                   num_time_steps - 1,
                                                   reporter);

    EvidenceSet test_data(data_dir);
    test_data.shrink_up_to(num_time_steps - 1);
    experimentation.evaluate_and_save(
        params_dir, num_folds, eval_dir, test_data, train_dir);
}

int main(int argc, char* argv[]) {
    string experiment_id;
    string model_dir;
    string params_dir;
    string train_dir;
    string data_dir;
    string eval_dir;
    string agent_json;
    string reporter_type;
    unsigned int num_time_steps;
    unsigned int num_folds;
    unsigned int num_jobs;

    po::options_description desc("Allowed options");
    desc.add_options()("help,h",
                       "This program evaluates a series of predictions or "
                       "inferences on a pre-trained model.")(
        "exp_id", po::value<string>(&experiment_id), "Experiment identifier.")(
        "model_dir",
        po::value<string>(&model_dir),
        "Directory where the agent's model definition is saved.")(
        "params_dir",
        po::value<string>(&params_dir)->default_value(""),
        "Directory where the pre-trained model's parameters are saved.")(
        "data_dir",
        po::value<string>(&data_dir)->required(),
        "Directory where the data (evidence) is located.")(
        "eval_dir",
        po::value<string>(&eval_dir)->required(),
        "Directory where the evaluation file and report (if "
        "requested) will be saved.")(
        "agent_json",
        po::value<string>(&agent_json)->required(),
        "Filepath of the json file containing definitions about the agent.")(
        "K",
        po::value<unsigned int>(&num_folds)->default_value(1)->required(),
        "Number of folds for evaluation using cross-validation. This assumes "
        "the model was pre-trained using the same number of folds here "
        "defined.")("T",
                    po::value<unsigned int>(&num_time_steps)
                        ->default_value(600)
                        ->required(),
                    "Number of time steps to unroll the model into.")(
        "jobs",
        po::value<unsigned int>(&num_jobs)->default_value(4),
        "Number of jobs used for multi-thread inference.")(
        "train_dir",
        po::value<string>(&train_dir),
        "Directory where data used for training is. This is only required if "
        "one of the agent's estimator computes frequency over the training "
        "set.")("reporter",
                po::value<string>(&reporter_type)->default_value(""),
                "asist_study_2\nasist_study_3");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    if (vm.count("help")) {
        cout << desc << endl;
        return 1;
    }

    evaluate(experiment_id,
             model_dir,
             params_dir,
             train_dir,
             data_dir,
             eval_dir,
             agent_json,
             reporter_type,
             (int)num_folds,
             (int)num_time_steps,
             (int)num_jobs);
}
