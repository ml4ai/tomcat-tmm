#include <string>

#include <boost/program_options.hpp>
#include <gsl/gsl_rng.h>

#include "experiments/Experimentation.h"
#include "experiments/TomcatTA3V2.h"
#include "pgm/EvidenceSet.h"

using namespace std;
using namespace tomcat::model;
namespace po = boost::program_options;

#define MODEL_VERSION Experimentation::MODEL_VERSION
#define MEASURES Experimentation::MEASURES

void train(const std::string& data_dir, const std::string& model_dir) {
    shared_ptr<gsl_rng> gen(gsl_rng_alloc(gsl_rng_mt19937));
    EvidenceSet training_set(data_dir);
    EvidenceSet test_set;
    Experimentation experimentation(gen,
                                    "training_study-1_2020.08",
                                    Experimentation::MODEL_VERSION::v2,
                                    training_set,
                                    test_set);
    experimentation.train_using_gibbs(1, 2);
    experimentation.save_model(model_dir, true);
    experimentation.train_and_save();
}

void evaluate(const std::string& data_dir,
              const std::string& model_dir,
              const std::string& eval_dir,
              unsigned int horizon) {
    shared_ptr<gsl_rng> gen(gsl_rng_alloc(gsl_rng_mt19937));
    EvidenceSet training_set;
    EvidenceSet test_set(data_dir);
    Experimentation experimentation(gen,
                                    "evaluation_study-1_2020.08",
                                    Experimentation::MODEL_VERSION::v2,
                                    training_set,
                                    test_set);
    experimentation.load_model_from(model_dir);

    vector<MEASURES> measures = {MEASURES::accuracy};
    experimentation.compute_eval_scores_for(TomcatTA3V2::Q, 0, measures);
    Eigen::VectorXd assignment = Eigen::VectorXd::Constant(1, 1);
    experimentation.compute_eval_scores_for(
        TomcatTA3V2::SG, horizon, measures, assignment);
    experimentation.compute_eval_scores_for(
        TomcatTA3V2::SY, horizon, measures, assignment);

    experimentation.display_estimates();
    experimentation.train_and_evaluate(eval_dir, true);
}

int main(int argc, char* argv[]) {
    int experiment_type;
    unsigned int horizon;
    string data_dir;
    string model_dir;
    string eval_dir;

    po::options_description desc("Allowed options");
    desc.add_options()("help,h", "Produce this help message")(
        "type",
        po::value<int>(&experiment_type)->default_value(0),
        "Experiment Type.\n"
        "  0: Training\n"
        "  1: Evaluation\n")(
        "data_dir",
        po::value<string>(&data_dir)->default_value(
            "../../data/asist/study-1_2020.08_split/train"),
        "Directory where input data is (training or evaluation).")(
        "model_dir",
        po::value<string>(&model_dir)
            ->default_value("../../data/model/asist/study-1_2020.08/"),
        "Directory where the model must be saved (training) or read from "
        "(evaluation).")(
        "eval_dir",
        po::value<string>(&eval_dir)->default_value(
            "../../data/eval/asist/study-1_2020.08/"),
        "Directory where the evaluation file should be saved.")(
        "horizon",
        po::value<unsigned int>(&horizon)->default_value(1),
        "Horizon of prediction for victim rescue in seconds.\n");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    if (vm.count("help") || !vm.count("type") || !vm.count("data_dir") ||
        !vm.count("model_dir")) {
        cout << desc << "\n";
        return 1;
    }


    if (experiment_type == 0) {
        train(data_dir, model_dir);
    }
    else if (experiment_type == 1) {
        if (!vm.count("eval_dir")) {
            cout << "A directory must be provided for saving evaluations.\n";
            return 1;
        }
        else if (horizon < 1) {
            cout << "The horizon of prediction has to be greater than zero.\n";
            return 1;
        }
        else {
            evaluate(data_dir, model_dir, eval_dir, horizon);
        }
    }
    else {
        cout << "Invalid experiment type.\n";
        return 1;
    }
}