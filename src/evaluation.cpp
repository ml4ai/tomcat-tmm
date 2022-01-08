#include <memory>
#include <string>
#include <vector>

#include <boost/program_options.hpp>
#include <fmt/format.h>
#include <gsl/gsl_rng.h>

#include "experiments/Experimentation.h"
#include "pgm/DynamicBayesNet.h"
#include "pgm/EvidenceSet.h"
#include "reporter/ASISTStudy2EstimateReporter.h"
#include "reporter/ASISTStudy3InterventionReporter.h"
#include "reporter/EstimateReporter.h"

using namespace tomcat::model;
using namespace std;
namespace po = boost::program_options;

struct ReporterTypes {
    const static int NONE = 0;
    const static int ASIST_STUDY2 = 1;
    const static int ASIST_STUDY3 = 2;
};

void evaluate(const string& experiment_id,
              const string& model_dir,
              const string& params_dir,
              const string& train_dir,
              const string& data_dir,
              const string& eval_dir,
              const string& agent_json,
              int num_folds,
              int num_time_steps,
              int num_particles,
              int num_jobs,
              bool baseline,
              bool exact_inference,
              int reporter_type,
              const string& report_filename) {

    shared_ptr<gsl_rng> random_generator(gsl_rng_alloc(gsl_rng_mt19937));

    string model_name;
    fstream file;
    file.open(agent_json);
    if (file.is_open()) {
        model_name = nlohmann::json::parse(file)["agent"]["model"];
    }

    string model_filepath = fmt::format("{}/{}.json", model_dir, model_name);
    shared_ptr<DynamicBayesNet> model = make_shared<DynamicBayesNet>(
        DynamicBayesNet ::create_from_json(model_filepath));
    model->unroll(3, true);

    Experimentation experimentation(random_generator, experiment_id, model);

    EstimateReporterPtr reporter;
    if (reporter_type == ReporterTypes::ASIST_STUDY2) {
        reporter = make_shared<ASISTStudy2EstimateReporter>();
    } else if (reporter_type == ReporterTypes::ASIST_STUDY3) {
        reporter = make_shared<ASISTStudy3InterventionReporter>();
    }
    string report_filepath;
    if (report_filename != "") {
        report_filepath = fmt::format("{}/{}", eval_dir, report_filename);
    }
    experimentation.set_offline_estimation_process(agent_json,
                                                   num_particles,
                                                   num_jobs,
                                                   baseline,
                                                   exact_inference,
                                                   num_time_steps - 1,
                                                   reporter,
                                                   report_filepath);

    EvidenceSet test_data(data_dir);
    test_data.shrink_up_to(num_time_steps - 1);
    test_data.keep_first(1);
    experimentation.evaluate_and_save(
        params_dir, num_folds, eval_dir, test_data, baseline, train_dir);
}

int main(int argc, char* argv[]) {
    string experiment_id;
    string model_dir;
    string params_dir;
    string train_dir;
    string data_dir;
    string eval_dir;
    string agent_json;
    string report_filename;
    unsigned int num_time_steps;
    unsigned int num_folds;
    unsigned int num_particles;
    unsigned int num_jobs;
    unsigned int reporter_type;
    bool baseline;
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
        "Directory where the evaluation file and estimate report (if "
        "requested) will be saved.")(
        "agent_json",
        po::value<string>(&agent_json)->required(),
        "Filepath of the json file containing definitions about the agents' "
        "reasoning")(
        "K",
        po::value<unsigned int>(&num_folds)->default_value(1)->required(),
        "Number of folds for evaluation using cross-validation. This assumes "
        "the model was pre-trained using the same number of folds here "
        "defined.")("T",
                    po::value<unsigned int>(&num_time_steps)
                        ->default_value(600)
                        ->required(),
                    "Number of time steps to unroll the model into.")(
        "particles",
        po::value<unsigned int>(&num_particles)->default_value(100)->required(),
        "Number of samples used to estimate the parameters of the model "
        "after the burn-in period if approximate inference is used for "
        "evaluation.")("jobs",
                       po::value<unsigned int>(&num_jobs)->default_value(4),
                       "Number of jobs used for multi-thread inference.")(
        "baseline",
        po::bool_switch(&baseline)->default_value(false),
        "Whether the baseline estimator based on frequencies of the samples in"
        " the training data must be used.")(
        "train_dir",
        po::value<string>(&train_dir),
        "Directory where data used for training is. This is only required for"
        " the baseline evaluation.")(
        "exact",
        po::bool_switch(&exact_inference)->default_value(false),
        "Whether to use exact or approximate inference.")(
        "reporter",
        po::value<unsigned int>(&reporter_type)->default_value(0)->required(),
        "0 - None\n"
        "1 - ASIST Study 2\n"
        "3 - ASIST Study 3")("report_filename",
                             po::value<string>(&report_filename),
                             "Filename of the reporter (if a reporter is "
                             "provided)..");

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
             model_dir,
             params_dir,
             train_dir,
             data_dir,
             eval_dir,
             agent_json,
             num_folds,
             num_time_steps,
             num_particles,
             num_jobs,
             baseline,
             exact_inference,
             reporter_type,
             report_filename);
}
