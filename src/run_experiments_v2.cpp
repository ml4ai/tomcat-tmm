/**
 * This source file implements the experiments described in details in the
 * document experimentation.pdf for the version 2.0 of the ToMCAT model.
 */

#include <string>
#include <vector>

#include <boost/program_options.hpp>
#include <eigen3/Eigen/Dense>
#include <fmt/format.h>
#include <gsl/gsl_rng.h>

#include "experiments/Experimentation.h"
#include "experiments/TomcatTA3V2.h"
#include "pgm/EvidenceSet.h"

using namespace tomcat::model;
using namespace std;
namespace po = boost::program_options;

#define MODEL_VERSION Experimentation::MODEL_VERSION
#define MEASURES Experimentation::MEASURES

string DATA_DIR;
string MODEL_DIR;
string EVAL_DIR;
string GEN_DATA_DIR;

/**
 * Performs a 10 cross validation on the falcon map using human data to predict
 * victim rescuing for several values of inference horizon.
 */
void execute_experiment_2a() {
    cout << "Experiment 2a\n";

    // Random Seed
    shared_ptr<gsl_rng> gen(gsl_rng_alloc(gsl_rng_mt19937));

    // Data
    EvidenceSet data(DATA_DIR);

    Experimentation experimentation(
        gen, "2a", Experimentation::MODEL_VERSION::v2, data, 10);

    experimentation.display_estimates();
    experimentation.train_using_gibbs(50, 100);
    string model_dir = fmt::format("{}/2a", MODEL_DIR);
    experimentation.save_model(model_dir, false);

    vector<int> horizons = {1, 3, 5, 10, 15, 30, 50, 100};
    Eigen::VectorXd assignment = Eigen::VectorXd::Constant(1, 1);
    vector<MEASURES> measures = {MEASURES::accuracy, MEASURES::f1};
    for (int horizon : horizons) {
        experimentation.compute_baseline_eval_scores_for(
            TomcatTA3::SG, horizon, measures, assignment);
        experimentation.compute_baseline_eval_scores_for(
            TomcatTA3::SY, horizon, measures, assignment);

        experimentation.compute_eval_scores_for(
            TomcatTA3::SG, horizon, measures, assignment);
        experimentation.compute_eval_scores_for(
            TomcatTA3::SY, horizon, measures, assignment);
    }

    string evaluations_dir = fmt::format("{}/2a", EVAL_DIR);
    experimentation.train_and_evaluate(evaluations_dir);
}

/**
 * Performs a 10 cross validation on the falcon map using human data to predict
 * the training condition used in each mission trial.
 */
void execute_experiment_2b() {
    cout << "Experiment 2b\n";

    // Random Seed
    shared_ptr<gsl_rng> gen(gsl_rng_alloc(gsl_rng_mt19937));

    // Data
    EvidenceSet data(DATA_DIR);

    Experimentation experimentation(
        gen, "2b", Experimentation::MODEL_VERSION::v2, data, 10);

    experimentation.train_using_gibbs(50, 100);
    string model_dir = fmt::format("{}/2b", MODEL_DIR);
    experimentation.save_model(model_dir, true);

    vector<MEASURES> measures = {MEASURES::accuracy};
    experimentation.compute_baseline_eval_scores_for(
        TomcatTA3V2::Q, 0, measures);
    experimentation.compute_eval_scores_for(TomcatTA3V2::Q, 0, measures);

    string evaluations_dir = fmt::format("{}/2b", EVAL_DIR);
    experimentation.display_estimates();
    experimentation.train_and_evaluate(evaluations_dir, true);
}

/**
 * Trains the model using the complete human data.
 */
void execute_experiment_2c_part_a() {
    cout << "Experiment 2c part a\n";

    // Random Seed
    shared_ptr<gsl_rng> gen(gsl_rng_alloc(gsl_rng_mt19937));

    // Data
    EvidenceSet training_set(DATA_DIR);
    EvidenceSet test_set;

    Experimentation experimentation(
        gen, "2c", Experimentation::MODEL_VERSION::v2, training_set, test_set);

    experimentation.train_using_gibbs(50, 100);
    string model_dir = fmt::format("{}/2c", MODEL_DIR);
    experimentation.save_model(model_dir, true);
    experimentation.train_and_save();
}

/**
 * Evaluates the inference of training condition on the model trained in part a
 * using the data used to train it.
 */
void execute_experiment_2c_part_b() {
    cout << "Experiment 2c part b\n";

    // Random Seed
    shared_ptr<gsl_rng> gen(gsl_rng_alloc(gsl_rng_mt19937));

    // Data
    EvidenceSet training_set(DATA_DIR);
    EvidenceSet test_set(DATA_DIR);

    Experimentation experimentation(
        gen, "2c", Experimentation::MODEL_VERSION::v2, training_set, test_set);

    experimentation.display_estimates();
    string model_dir = fmt::format("{}/2c", MODEL_DIR);
    experimentation.load_model_from(model_dir);

    vector<MEASURES> measures = {MEASURES::accuracy};
    experimentation.compute_baseline_eval_scores_for(
        TomcatTA3V2::Q, 0, measures);
    experimentation.compute_eval_scores_for(TomcatTA3V2::Q, 0, measures);

    string evaluations_dir = fmt::format("{}/2c", EVAL_DIR);
    experimentation.train_and_evaluate(evaluations_dir, true);
}

/**
 * Generates synthetic data from the model trained in 2c part a.
 */
void execute_experiment_2d_part_a() {
    cout << "Experiment 2d part a\n";

    // Random Seed
    shared_ptr<gsl_rng> gen(gsl_rng_alloc(gsl_rng_mt19937));

    Experimentation experimentation(gen, Experimentation::MODEL_VERSION::v2);

    string model_dir = fmt::format("{}/2c", MODEL_DIR);
    experimentation.load_model_from(model_dir);
    string samples_dir = fmt::format("{}/2d", GEN_DATA_DIR);
    experimentation.generate_synthetic_data(100, samples_dir);
}

/**
 * Performs a 10 cross validation on the falcon map using the synthetic data
 * generated in part a to predict the training condition used in each mission
 * trial.
 */
void execute_experiment_2d_part_b() {
    cout << "Experiment 2d part b\n";

    // Random Seed
    shared_ptr<gsl_rng> gen(gsl_rng_alloc(gsl_rng_mt19937));

    // Data
    EvidenceSet data(DATA_DIR);

    Experimentation experimentation(
        gen, "2d_cv", Experimentation::MODEL_VERSION::v2, data, 10);

    experimentation.train_using_gibbs(50, 100);
    string model_dir = fmt::format("{}/2d", MODEL_DIR);
    experimentation.save_model(model_dir, true);

    vector<MEASURES> measures = {MEASURES::accuracy};
    experimentation.compute_baseline_eval_scores_for(
        TomcatTA3V2::Q, 0, measures);
    experimentation.compute_eval_scores_for(TomcatTA3V2::Q, 0, measures);

    string evaluations_dir = fmt::format("{}/2d/cv", EVAL_DIR);
    experimentation.display_estimates();
    experimentation.train_and_evaluate(evaluations_dir, true);
}

/**
 * Evaluates the inference of training condition on the model trained in 2c part
 * a using the synthetic data generated from it in 2b part a.
 */
void execute_experiment_2d_part_c() {
    cout << "Experiment 2d part c\n";

    // Random Seed
    shared_ptr<gsl_rng> gen(gsl_rng_alloc(gsl_rng_mt19937));

    // Data
    string data_dir = fmt::format("{}/2d", GEN_DATA_DIR);
    EvidenceSet training_set(data_dir);
    EvidenceSet test_set(data_dir);

    Experimentation experimentation(
        gen, "2d", Experimentation::MODEL_VERSION::v2, training_set, test_set);

    string model_dir = fmt::format("{}/2c", MODEL_DIR);
    experimentation.load_model_from(model_dir);

    vector<MEASURES> measures = {MEASURES::accuracy};
    experimentation.compute_baseline_eval_scores_for(
        TomcatTA3V2::Q, 0, measures);
    experimentation.compute_eval_scores_for(TomcatTA3V2::Q, 0, measures);

    string evaluations_dir = fmt::format("{}/2d", EVAL_DIR);
    experimentation.display_estimates();
    experimentation.train_and_evaluate(evaluations_dir, true);
}

/**
 * Train in 60% of the data, save model and data sets.
 */
void execute_experiment_2e_part_a() {
    cout << "Experiment 2e part a\n";

    // Random Seed
    shared_ptr<gsl_rng> gen(gsl_rng_alloc(gsl_rng_mt19937));

    // Data
    EvidenceSet data_set(DATA_DIR);
    DataSplitter splitter(data_set, 0.4f, gen);
    EvidenceSet training_set = splitter.get_splits()[0].first;
    EvidenceSet validation_set = training_set;
    EvidenceSet test_set = splitter.get_splits()[0].second;

    training_set.save(fmt::format("{}/2e/training", GEN_DATA_DIR));
    test_set.save(fmt::format("{}/2e/test", GEN_DATA_DIR));

    Experimentation experimentation(
        gen, "2e", Experimentation::MODEL_VERSION::v2, training_set, test_set);

    experimentation.train_using_gibbs(50, 100);
    string model_dir = fmt::format("{}/2e", MODEL_DIR);
    experimentation.save_model(model_dir, false);
    experimentation.train_and_save();
}

/**
 * Evaluates the model on training and test data from the previous part.
 */
void execute_experiment_2e_part_b() {
    cout << "Experiment 2e part b\n";

    // Random Seed
    shared_ptr<gsl_rng> gen(gsl_rng_alloc(gsl_rng_mt19937));

    // Data
    EvidenceSet validation_set(fmt::format("{}/2e/training", GEN_DATA_DIR));
    EvidenceSet test_set(fmt::format("{}/2e/test", GEN_DATA_DIR));

    vector<EvidenceSet> evidence_sets = {validation_set, test_set};
    vector<string> evidence_id = {"val", "test"};
    int i = 0;

    for (auto& evidence_set : evidence_sets) {
        Experimentation experimentation(
            gen, "2e", Experimentation::MODEL_VERSION::v2, {}, evidence_set);

        string model_dir = fmt::format("{}/2e", MODEL_DIR);
        experimentation.load_model_from(model_dir);

        vector<MEASURES> measures = {MEASURES::accuracy};
        experimentation.compute_eval_scores_for(TomcatTA3V2::Q, 0, measures);

        string evaluations_dir =
            fmt::format("{}/2e_{}", EVAL_DIR, evidence_id[i++]);
        experimentation.display_estimates();
        experimentation.train_and_evaluate(evaluations_dir, false);
    }
}

/**
 * Performs a 10 cross validation on the falcon map using human data to predict
 * the confidence scale of the player in each mission trial.
 */
void execute_experiment_2f() {
    cout << "Experiment 2f\n";

    // Random Seed
    shared_ptr<gsl_rng> gen(gsl_rng_alloc(gsl_rng_mt19937));

    // Data
    EvidenceSet data(DATA_DIR);

    Experimentation experimentation(
        gen, "2f", Experimentation::MODEL_VERSION::v2, data, 10);

    experimentation.train_using_gibbs(50, 100);
    string model_dir = fmt::format("{}/2f", MODEL_DIR);
    experimentation.save_model(model_dir, false);

    vector<MEASURES> measures = {MEASURES::accuracy};
    experimentation.compute_baseline_eval_scores_for(
        TomcatTA3V2::Q, 0, measures);
    experimentation.compute_eval_scores_for(TomcatTA3V2::Q, 0, measures);

    string evaluations_dir = fmt::format("{}/2f", EVAL_DIR);
    experimentation.display_estimates();
    experimentation.train_and_evaluate(evaluations_dir, false);
}

void execute_experiment(const string& experiment_id) {
    if (experiment_id == "2a") {
        execute_experiment_2a();
    }
    else if (experiment_id == "2b") {
        execute_experiment_2b();
    }
    else if (experiment_id == "2c") {
        execute_experiment_2c_part_a();
        execute_experiment_2c_part_b();
    }
    else if (experiment_id == "2c_a") {
        execute_experiment_2c_part_a();
    }
    else if (experiment_id == "2c_b") {
        execute_experiment_2c_part_b();
    }
    else if (experiment_id == "2d") {
        execute_experiment_2d_part_a();
        execute_experiment_2d_part_b();
        execute_experiment_2d_part_c();
    }
    else if (experiment_id == "2d_a") {
        execute_experiment_2d_part_a();
    }
    else if (experiment_id == "2d_b") {
        execute_experiment_2d_part_b();
    }
    else if (experiment_id == "2d_c") {
        execute_experiment_2d_part_c();
    }
    else if (experiment_id == "2e") {
        execute_experiment_2e_part_a();
        execute_experiment_2e_part_b();
    }
    else if (experiment_id == "2e_a") {
        execute_experiment_2e_part_a();
    }
    else if (experiment_id == "2e_b") {
        execute_experiment_2e_part_b();
    }
    else if (experiment_id == "2f") {
        execute_experiment_2f();
    }
    else {
        throw TomcatModelException(
            "There's no experiment with the informed ID.");
    }
}

int main(int argc, char* argv[]) {
    string experiment_id;
    po::options_description desc("Allowed options");
    desc.add_options()("help,h", "Produce this help message")(
        "input_dir",
        po::value<string>(&DATA_DIR)->default_value("../../data/"),
        "Directory where the input data is.")(
        "model_dir",
        po::value<string>(&MODEL_DIR)->default_value("../../data/model"),
        "Directory where the model(s) must be saved or loaded (pre-trained).")(
        "output_dir",
        po::value<string>(&EVAL_DIR)->default_value("../../data/eval"),
        "Output directory for evaluation")(
        "gen_data_dir",
        po::value<string>(&GEN_DATA_DIR)->default_value("../../data/samples"),
        "Output directory for generated data.")(
        "experiment_id",
        po::value<string>(&experiment_id)->default_value("2a"),
        "Experiment ID.\n"
        "  2a: Evaluation of victim rescuing prediction using 10-cv for "
        "several "
        "horizons on human data.\n"
        "  2b: Evaluation of training condition inference using 10-cv on human "
        "data.\n"
        "  2c: Executes all parts of this experiment in sequence.\n"
        "  2c_a: Model training (and saving) using full human data.\n"
        "  2c_b: Evaluation of training condition inference on full human data "
        "using model trained in 2c_a.\n"
        "  2d: Executes all parts of this experiment in sequence.\n"
        "  2d_a: Synthetic data generation from the model trained in 2c_a.\n"
        "  2d_b: Evaluation of training condition inference using 10-cv on "
        "data generated in 2d_a.\n"
        "  2d_c: Evaluation of training condition inference on data generated "
        "in 2d_a using model trained in 2c_a.\n"
        "  2e: Executes all parts of this experiment in sequence.\n"
        "  2e_a: Model training on 60% of the data.\n"
        "  2e_b: Evaluation of training condition inference on training "
        "and test data.\n"
        "  2f: Evaluation of confidence scale inference using 10-cv on "
        "human data.\n");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    if (vm.count("help") || !vm.count("experiment_id")) {
        cout << desc << "\n";
        return 1;
    }

    execute_experiment(experiment_id);
}
