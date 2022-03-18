#include "Experimentation.h"

#include <boost/filesystem.hpp>
#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include "pipeline/DataSplitter.h"
#include "pipeline/Model.h"
#include "pipeline/Pipeline.h"
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
#include "reporter/EstimateReporter.h"
#include "sampling/GibbsSampler.h"
#include "utils/Definitions.h"
#include "utils/FileHandler.h"
#include "utils/JSONChecker.h"

namespace tomcat {
    namespace model {

        using namespace std;
        namespace fs = boost::filesystem;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        Experimentation::Experimentation(const shared_ptr<gsl_rng>& gen,
                                         const string& experiment_id,
                                         const shared_ptr<Model>& model)
            : random_generator(gen), experiment_id(experiment_id),
              model(model) {}

        Experimentation::~Experimentation() {}

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        void Experimentation::set_gibbs_trainer(int burn_in,
                                                int num_samples,
                                                int num_jobs) {
            if (auto dbn = dynamic_pointer_cast<DynamicBayesNet>(this->model)) {
                this->trainer = make_shared<DBNSamplingTrainer>(
                    this->random_generator,
                    make_shared<GibbsSampler>(dbn, burn_in, num_jobs),
                    num_samples);
            }
            else {
                throw TomcatModelException(
                    "A Gibbs trainer can only be used with PGM models.")
            }
        }

        void Experimentation::set_dbn_saver(const string& params_dir,
                                            int num_folds) {
            if (auto dbn = dynamic_pointer_cast<DynamicBayesNet>(this->model)) {
                if (this->trainer) {
                    string final_params_dir;
                    if (num_folds == 1) {
                        final_params_dir = params_dir;
                    }
                    else {
                        final_params_dir =
                            fmt::format("{}/fold{{}}", params_dir);
                    }

                    this->saver(dbn, this->trainer, final_params_dir, false);
                }
                else {
                    throw TomcatModelException("A DBN trainer must be set "
                                               "before a saver can be created.")
                }
            }
            else {
                throw TomcatModelException(
                    "A DBN saver can only be used with PGM models.");
            }
        }

        void Experimentation::train_and_save(const string& params_dir,
                                             int num_folds,
                                             const EvidenceSet& data) {
            if (!this->trainer) {
                throw TomcatModelException("A model trainer was not set.");
            }

            if (!this->saver) {
                throw TomcatModelException("A model saver was not set.");
            }

            DataSplitter splitter;
            if (num_folds == 1) {
                EvidenceSet empty_test_data;
                splitter = DataSplitter(data, empty_test_data);
            }
            else {
                try {
                    // Tries to load pre-existent indices from the params
                    // folder.
                    splitter = DataSplitter(data, params_dir);
                }
                catch (TomcatModelException& e) {
                    splitter =
                        DataSplitter(data, num_folds, this->random_generator);
                    splitter.save_indices(params_dir);
                }
            }

            int fold = 1;
            for (const auto& [training_data, test_data] :
                 splitter.get_splits()) {
                cout << "------------\n";
                cout << "Fold " << fold++ << endl;
                cout << "------------\n\n";

                this->trainer->prepare();
                this->trainer->fit(training_data);
                if (this->saver) {
                    this->saver->save();
                }
            }
        }

        void Experimentation::set_offline_estimation_process(
            const string& agent_config_filepath,
            int num_jobs,
            int max_time_step,
            const EstimateReporterPtr& estimate_reporter,
            const string& report_filepath) {

            AgentPtr agent = this->create_agent(
                agent_config_filepath, num_jobs, max_time_step);

            this->estimation = make_shared<OfflineEstimation>(
                agent, estimate_reporter, report_filepath);
        }

        void Experimentation::set_online_estimation_process(
            const string& agent_config_filepath,
            int num_jobs,
            int max_time_step,
            const string& message_broker_config_filepath,
            const MsgConverterPtr& converter,
            const EstimateReporterPtr& estimate_reporter) {

            AgentPtr agent = this->create_agent(
                agent_config_filepath, num_jobs, max_time_step);

            MessageBrokerConfiguration config;
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
                                      int num_jobs,
                                      int max_time_step) {

            AgentPtr agent;
            try {
                fstream file;
                file.open(agent_config_filepath);
                if (file.is_open()) {
                    auto json_object = nlohmann::json::parse(file);

                    check_field(json_object, "agent");

                    nlohmann::json json_agent =
                        nlohmann::json::parse(file)["agent"];

                    check_field(json_agent, "id");
                    check_field(json_agent, "version");

                    agent = make_shared<Agent>(json_agent["id"],
                                               json_agent["version"]);

                    unordered_set<string> ignored_observations;
                    if (EXISTS("ignored_observations", json_agent)) {
                        for (const auto& node_label :
                             json_agent["ignored_observations"]) {
                            ignored_observations.insert((string)node_label);
                        }
                        agent->set_ignored_observations(ignored_observations);
                    }

                    this->evaluation = make_shared<EvaluationAggregator>(
                        EvaluationAggregator::METHOD::no_aggregation);

                    shared_ptr<ParticleFilterEstimator>
                        particle_filter_estimator;

                    for (const auto& json_estimator :
                         json_agent["estimators"]) {

                        check_field(json_estimator, "type");
                        check_field(json_estimator, "name");
                        check_field(json_estimator, "settings");

                        auto& json_settings = json_estimator["settings"];

                        EstimatorPtr estimator;
                        if (json_estimator["type"] == "dbn") {
                            check_field(json_settings, "variable");
                            check_field(json_settings, "horizon");
                            check_field(json_settings, "value");

                            if (const auto& dbn =
                                    dynamic_pointer_cast<DynamicBayesNet>(
                                        this->model)) {

                                if (!dbn->has_node_with_label(
                                        json_estimator["variable"])) {
                                    throw TomcatModelException(fmt::format(
                                        "The variable {} does not belong to "
                                        "the model.",
                                        json_estimator["variable"]));
                                }

                                auto value = Eigen::VectorXd::Constant(
                                    1, stod((string)json_estimator["value"]));

                                if (json_estimator["name"] ==
                                    TrainingFrequencyEstimator::NAME) {
                                    estimator =
                                        make_shared<TrainingFrequencyEstimator>(
                                            dbn,
                                            json_estimator["horizon"],
                                            json_estimator["variable"],
                                            value);
                                }
                                else if (json_estimator["name"] ==
                                         SumProductEstimator::NAME) {
                                    estimator =
                                        make_shared<SumProductEstimator>(
                                            dbn,
                                            json_estimator["horizon"],
                                            json_estimator["variable"],
                                            value);
                                }
                                else if (json_estimator["name"] ==
                                         SamplerEstimator::NAME) {
                                    SamplerEstimator::FREQUENCY_TYPE
                                        estimation_frequency_type;
                                    unordered_set<int> fixed_time_steps;
                                    if (EXISTS("frequency", json_estimator)) {
                                        if (json_estimator["frequency"]
                                                          ["type"] == "fixed") {
                                            estimation_frequency_type =
                                                SamplerEstimator::fixed;
                                            const vector<int>& time_steps =
                                                json_estimator["frequency"]
                                                              ["time_steps"];
                                            fixed_time_steps =
                                                unordered_set<int>(
                                                    time_steps.begin(),
                                                    time_steps.end());
                                        }
                                        else if (json_estimator["frequency"]
                                                               ["type"] ==
                                                 "dynamic") {
                                            estimation_frequency_type =
                                                SamplerEstimator::dynamic;
                                        }
                                        else {
                                            estimation_frequency_type =
                                                SamplerEstimator::all;
                                        }
                                    }
                                    else {
                                        estimation_frequency_type =
                                            SamplerEstimator::all;
                                    }

                                    SamplerEstimatorPtr sampler_estimator =
                                        make_shared<SamplerEstimator>(
                                            dbn,
                                            json_estimator["horizon"],
                                            json_estimator["variable"],
                                            value,
                                            value,
                                            estimation_frequency_type);

                                    sampler_estimator->set_fixed_steps(
                                        fixed_time_steps);
                                    estimator = sampler_estimator;
                                }
                            }
                            else {
                                throw TomcatModelException(
                                    "DBN estimators are only "
                                    "defined for DBN models.");
                            }
                        }
                        else if (json_estimator["type"] == "custom") {
                            estimator = Estimator::factory(
                                json_estimator["name"], json_settings);
                        }

                        if (const auto& sampler_estimator =
                                dynamic_pointer_cast<SamplerEstimator>(
                                    estimator)) {
                            if (!particle_filter_estimator) {
                                check_field(json_agent, "num_particles");

                                particle_filter_estimator =
                                    make_shared<ParticleFilterEstimator>(
                                        dynamic_pointer_cast<DynamicBayesNet>(
                                            this->model),
                                        json_agent["num_particles"],
                                        this->random_generator,
                                        num_jobs,
                                        max_time_step);
                            }
                            particle_filter_estimator->add_base_estimator(
                                sampler_estimator);
                        }
                        else {
                            agent->add_estimator(estimator);
                        }

                        // Evaluation for the estimator
                        if (EXISTS("evaluation", json_estimator)) {
                            if (const auto& pgm_estimator =
                                    dynamic_pointer_cast<PGMEstimator>(
                                        estimator)) {
                                const vector<string>& measures =
                                    json_estimator["evaluation"]["measures"];
                                for (const auto& measure_name :
                                     unordered_set<string>(measures.begin(),
                                                           measures.end())) {

                                    Measure::FREQUENCY_TYPE eval_frequency_type;
                                    unordered_set<int> fixed_time_steps;
                                    if (json_estimator["evaluation"]
                                                      ["frequency"]["type"] ==
                                        "fixed") {
                                        eval_frequency_type = Measure::fixed;
                                        const vector<int> time_steps =
                                            json_estimator["evaluation"]
                                                          ["frequency"]
                                                          ["time_steps"];
                                        fixed_time_steps = unordered_set<int>(
                                            time_steps.begin(),
                                            time_steps.end());
                                    }
                                    else if (json_estimator["evaluation"]
                                                           ["frequency"]
                                                           ["type"] == "last") {
                                        eval_frequency_type = Measure::last;
                                    }
                                    else if (json_estimator["evaluation"]
                                                           ["frequency"]
                                                           ["type"] ==
                                             "dynamic") {
                                        eval_frequency_type = Measure::dynamic;
                                    }
                                    else {
                                        eval_frequency_type = Measure::all;
                                    }

                                    MeasurePtr measure;
                                    double thres = 0.5;
                                    if (measure_name == Accuracy::NAME) {
                                        measure = make_shared<Accuracy>(
                                            base_estimator,
                                            thres,
                                            eval_frequency_type);
                                    }
                                    else if (measure_name ==
                                             F1Score::MACRO_NAME) {
                                        measure = make_shared<F1Score>(
                                            base_estimator,
                                            thres,
                                            eval_frequency_type,
                                            true);
                                    }
                                    else if (measure_name ==
                                             F1Score::MICRO_NAME) {
                                        measure = make_shared<F1Score>(
                                            base_estimator,
                                            thres,
                                            eval_frequency_type,
                                            false);
                                    }
                                    else if (measure_name == RMSE::NAME) {
                                        measure = make_shared<RMSE>(
                                            base_estimator,
                                            eval_frequency_type);
                                    }

                                    measure->set_fixed_steps(fixed_time_steps);
                                    this->evaluation->add_measure(measure);
                                }
                            } else {
                                throw TomcatModelException("Currently, evaluations are only supported for PGM estimators.");
                            }
                        }
                    }

                    if (particle_filter_estimator) {
                        agent->add_estimator(particle_filter_estimator);
                    }

                    file.close();
                }
                else {
                    stringstream ss;
                    ss << "The file " << agent_config_filepath
                       << " does not exist.";
                    throw TomcatModelException(ss.str());
                }
            }
            catch (TomcatModelException& tom_ex) {
                throw TomcatModelException(fmt::format(
                    "Error while creating the agent. {}", tom_ex.message));
            }
            catch (exception& ex) {
                throw TomcatModelException(fmt::format(
                    "Error while creating the agent. {}", ex.what()));
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

            fs::create_directories(eval_dir);
            string filepath =
                fmt::format("{}/{}.json", eval_dir, this->experiment_id);
            ofstream output_file;
            output_file.open(filepath);

            this->estimation->set_display_estimates(true);
            Pipeline pipeline(this->experiment_id, output_file);
            pipeline.set_data_splitter(data_splitter);

            if (params_dir != "") {
                shared_ptr<DBNTrainer> loader =
                    make_shared<DBNLoader>(this->model, final_params_dir, true);
                pipeline.set_model_trainer(loader);
            }
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

            this->estimation->get_agent()->show_progress(false);

            Pipeline pipeline;
            pipeline.set_data_splitter(data_splitter);
            if (!params_dir.empty()) {
                shared_ptr<DBNTrainer> loader =
                    make_shared<DBNLoader>(this->model, params_dir, true);
                pipeline.set_model_trainer(loader);
            }
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

            if (params_dir != "") {
                shared_ptr<DBNTrainer> loader =
                    make_shared<DBNLoader>(this->model, params_dir, true);
                loader->fit({});
            }

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
