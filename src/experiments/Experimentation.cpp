#include "Experimentation.h"

#include <boost/filesystem.hpp>
#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include "pipeline/DBNSaver.h"
#include "pipeline/DataSplitter.h"
#include "pipeline/Pipeline.h"
#include "pipeline/estimation/EstimateReporter.h"
#include "pipeline/estimation/OfflineEstimation.h"
#include "pipeline/estimation/OnlineEstimation.h"
#include "pipeline/estimation/ParticleFilterEstimator.h"
#include "pipeline/estimation/SamplerEstimator.h"
#include "pipeline/estimation/SumProductEstimator.h"
#include "pipeline/estimation/TrainingFrequencyEstimator.h"
#include "pipeline/evaluation/Accuracy.h"
#include "pipeline/evaluation/F1Score.h"
#include "pipeline/evaluation/RMSE.h"
#include "pipeline/training/DBNLoader.h"
#include "pipeline/training/DBNSamplingTrainer.h"
#include "sampling/GibbsSampler.h"
#include "utils/Definitions.h"
#include "utils/FileHandler.h"

namespace tomcat {
    namespace model {

        using namespace std;
        namespace fs = boost::filesystem;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        Experimentation::Experimentation(
            const shared_ptr<gsl_rng>& gen,
            const string& experiment_id,
            const shared_ptr<DynamicBayesNet>& model)
            : random_generator(gen), experiment_id(experiment_id),
              model(model) {}

        Experimentation::~Experimentation() {}

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        void Experimentation::set_gibbs_trainer(int burn_in,
                                                int num_samples,
                                                int num_jobs) {
            this->trainer = make_shared<DBNSamplingTrainer>(
                this->random_generator,
                make_shared<GibbsSampler>(this->model, burn_in, num_jobs),
                num_samples);
        }

        void Experimentation::train_and_save(const string& params_dir,
                                             int num_folds,
                                             const EvidenceSet& data) {
            string final_params_dir;
            DataSplitter splitter;
            if (num_folds == 1) {
                EvidenceSet empty_test_data;
                splitter = DataSplitter(data, empty_test_data);
                final_params_dir = params_dir;
            }
            else {
                splitter =
                    DataSplitter(data, num_folds, this->random_generator);
                final_params_dir = fmt::format("{}/fold{{}}", params_dir);
                splitter.save_indices(params_dir);
            }

            DBNSaver model_saver(
                this->model, this->trainer, final_params_dir, false);

            int fold = 1;
            for (const auto& [training_data, test_data] :
                 splitter.get_splits()) {
                cout << "------------\n";
                cout << "Fold " << fold++ << endl;
                cout << "------------\n\n";

                this->trainer->prepare();
                this->trainer->fit(training_data);
                model_saver.save();
            }
        }

        void Experimentation::set_offline_estimation_process(
            const string& agent_config_filepath,
            int num_particles,
            int num_jobs,
            bool baseline,
            bool exact_inference,
            int max_time_step,
            const EstimateReporterPtr& estimate_reporter,
            const string& report_filepath) {

            AgentPtr agent = this->create_agent(agent_config_filepath,
                                                num_particles,
                                                num_jobs,
                                                baseline,
                                                exact_inference,
                                                max_time_step);

            this->estimation = make_shared<OfflineEstimation>(
                agent, estimate_reporter, report_filepath);
        }

        void Experimentation::set_online_estimation_process(
            const string& agent_config_filepath,
            int num_particles,
            int num_jobs,
            bool baseline,
            bool exact_inference,
            int max_time_step,
            const string& message_broker_config_filepath,
            const MsgConverterPtr& converter,
            const EstimateReporterPtr& estimate_reporter) {

            AgentPtr agent = this->create_agent(agent_config_filepath,
                                                num_particles,
                                                num_jobs,
                                                baseline,
                                                exact_inference,
                                                max_time_step);

            OnlineEstimation::MessageBrokerConfiguration config;
            config.timeout = INT32_MAX;
            fstream file;
            file.open(message_broker_config_filepath);
            if (file.is_open()) {
                nlohmann::json broker = nlohmann::json::parse(file);
                config.address = broker["address"];
                config.port = broker["port"];
                config.estimates_topic = broker["estimates_topic"];
                config.log_topic = broker["log_topic"];
                config.num_connection_trials = broker["num_connection_trials"];
                config.milliseconds_before_retrial =
                    broker["milliseconds_before_connection_retrial"];
            }

            this->estimation = make_shared<OnlineEstimation>(
                agent, config, converter, estimate_reporter);
        }

        AgentPtr
        Experimentation::create_agent(const string& agent_config_filepath,
                                      int num_particles,
                                      int num_jobs,
                                      bool baseline,
                                      bool exact_inference,
                                      int max_time_step) {

            AgentPtr agent;
            fstream file;
            file.open(agent_config_filepath);
            if (file.is_open()) {
                nlohmann::json json_agent =
                    nlohmann::json::parse(file)["agent"];

                if (json_agent.empty()) {
                    stringstream ss;
                    ss << "No agent to experiment with. The file "
                       << agent_config_filepath << "is empty.";
                    throw TomcatModelException(ss.str());
                }

                this->evaluation = make_shared<EvaluationAggregator>(
                    EvaluationAggregator::METHOD::no_aggregation);

                shared_ptr<ParticleFilterEstimator> approximate_estimator;
                approximate_estimator =
                    make_shared<ParticleFilterEstimator>(this->model,
                                                         num_particles,
                                                         this->random_generator,
                                                         num_jobs,
                                                         max_time_step);

                agent =
                    make_shared<Agent>(json_agent["id"], json_agent["version"]);
                unordered_set<string> ignored_observations;
                for (const string& node_label :
                     json_agent["ignored_observations"]) {
                    ignored_observations.insert(node_label);
                }
                agent->set_ignored_observations(ignored_observations);

                for (const auto& json_estimator : json_agent["estimators"]) {

                    SamplerEstimator::FREQUENCY_TYPE estimation_frequency_type;
                    unordered_set<int> fixed_time_steps;
                    if (EXISTS("frequency", json_estimator)) {
                        if (json_estimator["frequency"]["type"] == "fixed") {
                            estimation_frequency_type = SamplerEstimator::fixed;
                            fixed_time_steps = unordered_set<int>(
                                json_estimator["frequency"]["time_steps"]);
                        }
                        else if (json_estimator["frequency"]["type"] ==
                                 "dynamic") {
                            estimation_frequency_type =
                                SamplerEstimator::dynamic;
                        }
                        else {
                            estimation_frequency_type = SamplerEstimator::all;
                        }
                    }
                    else {
                        estimation_frequency_type = SamplerEstimator::all;
                    }

                    EstimatorPtr base_estimator;
                    if (baseline) {
                        if (json_estimator["type"] == "custom") {
                            // Not supported yet.
                        }
                        else {
                            if (!this->model->has_node_with_label(
                                    json_estimator["variable"])) {
                                stringstream ss;
                                ss << "The variable "
                                   << json_estimator["variable"]
                                   << "does not belong to the model";
                                throw TomcatModelException(ss.str());
                            }

                            Eigen::VectorXd value(0);
                            if (json_estimator["value"] != "") {
                                value = Eigen::VectorXd::Constant(
                                    1, stod((string)json_estimator["value"]));
                            }

                            EstimatorPtr estimator =
                                make_shared<TrainingFrequencyEstimator>(
                                    this->model,
                                    json_estimator["horizon"],
                                    json_estimator["variable"],
                                    value);

                            agent->add_estimator(estimator);
                        }
                    }
                    else {
                        if (json_estimator["type"] == "custom") {
                            SamplerEstimatorPtr estimator =
                                SamplerEstimator::create_custom_estimator(
                                    json_estimator["name"],
                                    this->model,
                                    estimation_frequency_type);

                            estimator->set_fixed_steps(fixed_time_steps);
                            approximate_estimator->add_base_estimator(
                                estimator);
                            base_estimator = estimator;
                        }
                        else {
                            if (!this->model->has_node_with_label(
                                    json_estimator["variable"])) {
                                stringstream ss;
                                ss << "The variable "
                                   << json_estimator["variable"]
                                   << "does not belong to the model";
                                throw TomcatModelException(ss.str());
                            }

                            Eigen::VectorXd value(0);
                            if (json_estimator["value"] != "") {
                                value = Eigen::VectorXd::Constant(
                                    1, stod((string)json_estimator["value"]));
                            }

                            if (exact_inference) {
                                base_estimator =
                                    make_shared<SumProductEstimator>(
                                        this->model,
                                        json_estimator["horizon"],
                                        json_estimator["variable"],
                                        value);

                                agent->add_estimator(base_estimator);
                            }
                            else {
                                SamplerEstimatorPtr estimator =
                                    make_shared<SamplerEstimator>(
                                        this->model,
                                        json_estimator["horizon"],
                                        json_estimator["variable"],
                                        value,
                                        value,
                                        estimation_frequency_type);

                                estimator->set_fixed_steps(fixed_time_steps);
                                approximate_estimator->add_base_estimator(
                                    estimator);
                                base_estimator = estimator;
                            }
                        }
                    }

                    // Evaluation for the estimator
                    if (EXISTS("evaluation", json_estimator)) {
                        for (const auto& measure_name : unordered_set<string>(
                                 json_estimator["evaluation"]["measures"])) {

                            Measure::FREQUENCY_TYPE eval_frequency_type;
                            unordered_set<int> fixed_time_steps;
                            if (json_estimator["evaluation"]["frequency"]
                                              ["type"] == "fixed") {
                                eval_frequency_type = Measure::fixed;
                                fixed_time_steps = unordered_set<int>(
                                    json_estimator["evaluation"]["frequency"]
                                                  ["time_steps"]);
                            }
                            else if (json_estimator["evaluation"]["frequency"]
                                                   ["type"] == "last") {
                                eval_frequency_type = Measure::last;
                            }
                            else if (json_estimator["evaluation"]["frequency"]
                                                   ["type"] == "dynamic") {
                                eval_frequency_type = Measure::dynamic;
                            }
                            else {
                                eval_frequency_type = Measure::all;
                            }

                            MeasurePtr measure;
                            if (measure_name == Accuracy::NAME) {
                                measure = make_shared<Accuracy>(
                                    base_estimator, 0.5, eval_frequency_type);
                            }
                            else if (measure_name == F1Score::MACRO_NAME) {
                                measure =
                                    make_shared<F1Score>(base_estimator,
                                                         0.5,
                                                         eval_frequency_type,
                                                         true);
                            }
                            else if (measure_name == F1Score::MICRO_NAME) {
                                measure =
                                    make_shared<F1Score>(base_estimator,
                                                         0.5,
                                                         eval_frequency_type,
                                                         false);
                            }
                            else if (measure_name == RMSE::NAME) {
                                measure = make_shared<RMSE>(
                                    base_estimator, eval_frequency_type);
                            }

                            measure->set_fixed_steps(fixed_time_steps);
                            this->evaluation->add_measure(measure);
                        }
                    }
                }

                if (!approximate_estimator->get_base_estimators().empty()) {
                    agent->add_estimator(approximate_estimator);
                }

                file.close();
            }
            else {
                stringstream ss;
                ss << "The file " << agent_config_filepath
                   << " does not exist.";
                throw TomcatModelException(ss.str());
            }

            return agent;
        }

        void Experimentation::evaluate_and_save(const string& params_dir,
                                                int num_folds,
                                                const string& eval_dir,
                                                const EvidenceSet& data,
                                                bool baseline,
                                                const string& train_dir) {
            shared_ptr<DataSplitter> data_splitter;
            string final_params_dir;
            if (num_folds > 1) {
                // One set of learned parameters per fold
                final_params_dir = fmt::format("{}/fold{{}}", params_dir);
                data_splitter = make_shared<DataSplitter>(data, params_dir);
            }
            else {
                final_params_dir = params_dir;
                EvidenceSet empty_set;
                if (baseline) {
                    // The baseline method outputs probabilities as the
                    // frequencies in the values of the samples used for
                    // training. If this executable is called with baseline
                    // set, we assume that the content of the parameter data
                    // is the training data.
                    data_splitter = make_shared<DataSplitter>(train_dir, data);
                }
                else {
                    data_splitter = make_shared<DataSplitter>(empty_set, data);
                }
            }

            shared_ptr<DBNTrainer> loader =
                make_shared<DBNLoader>(this->model, final_params_dir, true);
            EvidenceSet empty_training;

            fs::create_directories(eval_dir);
            string filepath =
                fmt::format("{}/{}.json", eval_dir, this->experiment_id);
            ofstream output_file;
            output_file.open(filepath);

            this->estimation->set_display_estimates(true);
            Pipeline pipeline(this->experiment_id, output_file);
            pipeline.set_data_splitter(data_splitter);
            pipeline.set_model_trainer(loader);
            pipeline.set_estimation_process(this->estimation);
            if (this->evaluation->has_measures()) {
                pipeline.set_aggregator(this->evaluation);
            }
            pipeline.execute();
            output_file.close();
        }

        void Experimentation::start_real_time_estimation(
            const std::string& params_dir) {
            // Data comes from the message bus
            EvidenceSet empty_set;
            shared_ptr<DataSplitter> data_splitter =
                make_shared<DataSplitter>(empty_set, empty_set);

            shared_ptr<DBNTrainer> loader =
                make_shared<DBNLoader>(this->model, params_dir, true);
            EvidenceSet empty_training;

            for (const auto& estimator :
                 this->estimation->get_agent()->get_estimators()) {
                for (const auto& base_estimator :
                     estimator->get_base_estimators()) {
                    estimator->set_show_progress(false);
                }
            }

            Pipeline pipeline;
            pipeline.set_data_splitter(data_splitter);
            pipeline.set_model_trainer(loader);
            pipeline.set_estimation_process(this->estimation);
            pipeline.execute();
        }

        void Experimentation::generate_synthetic_data(
            const string& params_dir,
            const string& data_dir,
            int num_data_samples,
            int num_time_steps,
            int equal_samples_time_step_limit,
            const unordered_set<string>& exclusions,
            int num_jobs) {

            this->model->unroll(num_time_steps, true);
            shared_ptr<DBNTrainer> loader =
                make_shared<DBNLoader>(this->model, params_dir, true);
            loader->fit({});

            AncestralSampler sampler(this->model, num_jobs);
            sampler.set_num_in_plate_samples(num_data_samples);
            sampler.set_equal_samples_time_step_limit(
                equal_samples_time_step_limit);
            sampler.sample(this->random_generator, num_data_samples);
            sampler.save_samples_to_folder(data_dir, exclusions);
        }

        void Experimentation::print_model(const std::string& params_dir,
                                          const std::string& model_dir) {

            this->model->unroll(3, true);
            shared_ptr<DBNTrainer> loader =
                make_shared<DBNLoader>(this->model, params_dir, true);
            loader->fit({});

            fs::create_directories(model_dir);
            ofstream output_file;
            string graph_filepath = get_filepath(model_dir, "graph.viz");
            output_file.open(graph_filepath);
            this->model->print_graph(output_file);
            output_file.close();

            if (params_dir != "") {
                string cpds_filepath = get_filepath(model_dir, "cpds.txt");
                output_file.open(cpds_filepath);
                this->model->print_cpds(output_file);
                output_file.close();
            }
        }

    } // namespace model
} // namespace tomcat
