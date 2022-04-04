#include "Experimentation.h"

#include <boost/filesystem.hpp>
#include <fmt/format.h>
#include <gsl/gsl_errno.h>
#include <nlohmann/json.hpp>

#include "pipeline/DataSplitter.h"
#include "pipeline/Model.h"
#include "pipeline/Pipeline.h"
#include "pipeline/estimation/OfflineEstimation.h"
#include "pipeline/estimation/OnlineEstimation.h"
#include "pipeline/estimation/OnlineLogger.h"
#include "pipeline/estimation/ParticleFilterEstimator.h"
#include "pipeline/estimation/SamplerEstimator.h"
#include "pipeline/estimation/SumProductEstimator.h"
#include "pipeline/estimation/TrainingFrequencyEstimator.h"
#include "pipeline/evaluation/Accuracy.h"
#include "pipeline/evaluation/F1Score.h"
#include "pipeline/evaluation/RMSE.h"
#include "pipeline/training/DBNLoader.h"
#include "pipeline/training/DBNSamplingTrainer.h"
#include "pipeline/training/ModelLoader.h"
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
                                         const string& experiment_id)
            : random_generator(gen), experiment_id(experiment_id) {
            gsl_set_error_handler_off();
        }

        Experimentation::~Experimentation() {}

        //----------------------------------------------------------------------
        // Static functions
        //----------------------------------------------------------------------

        Estimator::FREQUENCY_TYPE Experimentation::get_frequency_type(
            const nlohmann::json& json_frequency) {
            Estimator::FREQUENCY_TYPE frequency_type;

            if (json_frequency["type"] == "fixed") {
                frequency_type = Estimator::fixed;
            }
            else if (json_frequency["type"] == "last") {
                frequency_type = Estimator::last;
            }
            else if (json_frequency["type"] == "dynamic") {
                frequency_type = Estimator::dynamic;
            }
            else {
                frequency_type = Estimator::all;
            }

            return frequency_type;
        }

        std::unordered_set<int> Experimentation::get_fixed_time_steps(
            const nlohmann::json& json_frequency) {
            unordered_set<int> fixed_time_steps;
            if (json_frequency["type"] == "fixed") {
                const vector<int> time_steps = json_frequency["time_steps"];
                fixed_time_steps =
                    unordered_set<int>(time_steps.begin(), time_steps.end());
            }

            return fixed_time_steps;
        }

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
                    "A Gibbs trainer can only be used with PGM models.");
            }
        }

        void Experimentation::set_model_saver(const string& params_dir,
                                              int num_folds) {
            string final_params_dir;
            if (num_folds == 1) {
                final_params_dir = params_dir;
            }
            else {
                final_params_dir = fmt::format("{}/fold{{}}", params_dir);
            }

            if (auto dbn = dynamic_pointer_cast<DynamicBayesNet>(this->model)) {
                if (this->trainer) {
                    if (auto dbn_trainer =
                            dynamic_pointer_cast<DBNTrainer>(this->trainer)) {
                        this->saver = make_shared<DBNSaver>(
                            dbn, final_params_dir, dbn_trainer, false);
                    }
                    else {
                        throw TomcatModelException(
                            "A DBN saver can only work with a DBN trainer.");
                    }
                }
                else {
                    throw TomcatModelException(
                        "A DBN trainer must be set "
                        "before a saver can be created.");
                }
            }
            else {
                this->saver = make_shared<ModelSaver>(model, final_params_dir);
            }
        }

        void Experimentation::train_and_save(const string& params_dir,
                                             int num_folds,
                                             const EvidenceSet& data) {

            if (!this->trainer) {
                throw TomcatModelException("A model trainer was not set.");
            }

            this->set_model_saver(params_dir, num_folds);

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
                catch (const TomcatModelException& e) {
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
            const string& model_dir,
            const string& reporter_dir,
            int num_jobs,
            int max_time_step,
            const EstimateReporterPtr& estimate_reporter) {

            AgentPtr agent = this->create_agent(
                agent_config_filepath, model_dir, num_jobs, max_time_step);

            string reporter_filepath = fmt::format(
                "{}/{}_reporter.json", reporter_dir, this->experiment_id);

            this->estimation = make_shared<OfflineEstimation>(
                agent, estimate_reporter, reporter_filepath);
        }

        void Experimentation::set_online_estimation_process(
            const string& agent_config_filepath,
            const std::string& model_dir,
            int num_jobs,
            int max_time_step,
            const string& message_broker_config_filepath,
            const MsgConverterPtr& converter,
            const EstimateReporterPtr& estimate_reporter,
            const string& log_dir) {

            AgentPtr agent = this->create_agent(
                agent_config_filepath, model_dir, num_jobs, max_time_step);

            MessageBrokerConfiguration config;
            config.timeout = INT32_MAX;
            fstream file;
            file.open(message_broker_config_filepath);
            if (file.is_open()) {
                nlohmann::json broker = nlohmann::json::parse(file);
                config.address = broker["address"];
                config.port = broker["port"];
                config.intervention_topic = broker["estimates_topic"];
                config.num_connection_trials = broker["num_connection_trials"];
                config.milliseconds_before_retrial =
                    broker["milliseconds_before_connection_retrial"];
            }

            OnlineLoggerPtr logger;
            if (!log_dir.empty()) {
                fs::create_directories(log_dir);
                string log_filepath =
                    fmt::format("{}/{}.txt", log_dir, this->experiment_id);
                logger = make_shared<OnlineLogger>(log_filepath);
                estimate_reporter->set_logger(logger);
                for (auto& estimator : agent->get_estimators()) {
                    estimator->set_logger(logger);
                }
            }

            this->estimation = make_shared<OnlineEstimation>(
                agent, config, converter, estimate_reporter, logger);
        }

        AgentPtr
        Experimentation::create_agent(const string& agent_config_filepath,
                                      const string& model_dir,
                                      int num_jobs,
                                      int max_time_step) {

            AgentPtr agent;
            try {
                fstream file;
                file.open(agent_config_filepath);
                if (file.is_open()) {
                    auto json_object = nlohmann::json::parse(file);

                    check_field(json_object, "agent");

                    nlohmann::json json_agent = json_object["agent"];

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

                    check_field(json_agent, "model");

                    this->parse_model(json_agent["model"], model_dir);

                    check_field(json_agent, "estimators");

                    this->evaluation = make_shared<EvaluationAggregator>(
                        EvaluationAggregator::METHOD::no_aggregation);

                    ParticleFilterEstimatorPtr particle_filter_estimator;
                    if (EXISTS("num_particles", json_agent)) {
                        particle_filter_estimator =
                            make_shared<ParticleFilterEstimator>(
                                dynamic_pointer_cast<DynamicBayesNet>(
                                    this->model),
                                json_agent["num_particles"],
                                this->random_generator,
                                num_jobs,
                                max_time_step);
                    }

                    this->parse_estimators(json_agent["estimators"],
                                           agent,
                                           particle_filter_estimator,
                                           num_jobs);

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

        void Experimentation::parse_model(const nlohmann::json& json_model,
                                          const string& model_dir) {
            try {
                check_field(json_model, "type");
                check_field(json_model, "filename");

                string filename = json_model["filename"];
                string filepath;

                if (filename.empty() || model_dir.empty()) {
                    throw TomcatModelException(
                        "Model filename or directory not provided.");
                }
                else {
                    filepath = fmt::format("{}/{}", model_dir, filename);
                }

                if (json_model["type"] == "dbn") {
                    auto dbn = make_shared<DynamicBayesNet>(
                        DynamicBayesNet ::create_from_json(filepath));
                    dbn->unroll(3, true);
                    this->model = dbn;
                }
                else if (json_model["type"] == "custom") {
                    check_field(json_model, "name");

                    fstream file;
                    file.open(filepath);
                    if (file.is_open()) {
                        nlohmann::json json_settings =
                            nlohmann::json::parse(file);
                        this->model = Model::factory((string)json_model["name"],
                                                     json_settings);
                    }
                    else {
                        throw TomcatModelException(
                            fmt::format("File {} not found.", filepath));
                    }
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

            //    string model_name;
            //    fstream file;
            //    file.open(agent_json);
            //    if (file.is_open()) {
            //        model_name =
            //        nlohmann::json::parse(file)["agent"]["model"];
            //    }
            //
            //    string model_filepath = fmt::format("{}/{}.json", model_dir,
            //    model_name); shared_ptr<DynamicBayesNet> model =
            //    make_shared<DynamicBayesNet>(
            //        DynamicBayesNet ::create_from_json(model_filepath));
            //    model->unroll(3, true);
        }

        void Experimentation::parse_estimators(
            const nlohmann::json& json_estimators,
            const AgentPtr& agent,
            const ParticleFilterEstimatorPtr& particle_filter_estimator,
            int num_jobs) {

            for (const auto& json_estimator : json_estimators) {
                check_field(json_estimator, "type");
                check_field(json_estimator, "name");
                check_field(json_estimator, "settings");

                auto& json_settings = json_estimator["settings"];

                Estimator::FREQUENCY_TYPE frequency_type =
                    SamplerEstimator::all;
                unordered_set<int> fixed_time_steps;

                if (EXISTS("frequency", json_estimator)) {
                    frequency_type = Experimentation::get_frequency_type(
                        json_estimator["frequency"]);
                    fixed_time_steps = Experimentation::get_fixed_time_steps(
                        json_estimator["frequency"]);
                }

                EstimatorPtr estimator;
                if (json_estimator["type"] == "dbn") {
                    check_field(json_settings, "variable");
                    check_field(json_settings, "horizon");
                    check_field(json_settings, "value");

                    if (const auto& dbn = dynamic_pointer_cast<DynamicBayesNet>(
                            this->model)) {

                        if (!dbn->has_node_with_label(
                                json_settings["variable"])) {
                            throw TomcatModelException(fmt::format(
                                "The variable {} does not belong to "
                                "the model.",
                                (string)json_settings["variable"]));
                        }

                        Eigen::VectorXd value;
                        string value_str = (string)json_settings["value"];
                        if (!value_str.empty()) {
                            value = Eigen::VectorXd::Constant(
                                1, stod((string)json_settings["value"]));
                        }

                        if (json_estimator["name"] ==
                            TrainingFrequencyEstimator::NAME) {
                            estimator = make_shared<TrainingFrequencyEstimator>(
                                dbn,
                                json_settings["horizon"],
                                json_settings["variable"],
                                value);
                        }
                        else if (json_estimator["name"] ==
                                 SumProductEstimator::NAME) {
                            estimator = make_shared<SumProductEstimator>(
                                dbn,
                                json_settings["horizon"],
                                json_settings["variable"],
                                value);
                        }
                        else if (json_estimator["name"] ==
                                 SamplerEstimator::NAME) {

                            SamplerEstimatorPtr sampler_estimator =
                                make_shared<SamplerEstimator>(
                                    dbn,
                                    json_settings["horizon"],
                                    json_settings["variable"],
                                    value,
                                    value,
                                    frequency_type);

                            sampler_estimator->set_fixed_steps(
                                fixed_time_steps);
                            estimator = sampler_estimator;
                        }
                    }
                    else {
                        throw TomcatModelException("DBN estimators are only "
                                                   "defined for DBN models.");
                    }
                }
                else if (json_estimator["type"] == "custom") {
                    estimator = Estimator::factory(json_estimator["name"],
                                                   this->model,
                                                   json_settings,
                                                   frequency_type,
                                                   fixed_time_steps,
                                                   num_jobs);
                }

                if (const auto& sampler_estimator =
                        dynamic_pointer_cast<SamplerEstimator>(estimator)) {
                    if (particle_filter_estimator) {
                        particle_filter_estimator->add_base_estimator(
                            sampler_estimator);
                    }
                    else {
                        throw TomcatModelException(
                            "No particle filter created. Please, inform the "
                            "number of particles to generate for estimation in "
                            "the agent's definition.");
                    }
                }
                else {
                    agent->add_estimator(estimator);
                }

                if (particle_filter_estimator &&
                    !particle_filter_estimator->get_base_estimators().empty()) {
                    agent->add_estimator(particle_filter_estimator);
                }

                if (EXISTS("evaluation", json_estimator)) {
                    this->parse_evaluation(json_estimator["evaluation"],
                                           estimator);
                }
            }
        }

        void
        Experimentation::parse_evaluation(const nlohmann::json& json_evaluation,
                                          const EstimatorPtr& estimator) {
            if (const auto& pgm_estimator =
                    dynamic_pointer_cast<PGMEstimator>(estimator)) {
                const vector<string>& measures = json_evaluation["measures"];
                for (const auto& measure_name :
                     unordered_set<string>(measures.begin(), measures.end())) {

                    Estimator::FREQUENCY_TYPE frequency_type;
                    unordered_set<int> fixed_time_steps;
                    if (EXISTS("frequency", json_evaluation)) {
                        frequency_type = Experimentation::get_frequency_type(
                            json_evaluation["frequency"]);
                        fixed_time_steps =
                            Experimentation::get_fixed_time_steps(
                                json_evaluation["frequency"]);
                    }

                    MeasurePtr measure;
                    double threshold = 0.5;
                    if (measure_name == Accuracy::NAME) {
                        measure = make_shared<Accuracy>(
                            pgm_estimator, threshold, frequency_type);
                    }
                    else if (measure_name == F1Score::MACRO_NAME) {
                        measure = make_shared<F1Score>(
                            pgm_estimator, threshold, frequency_type, true);
                    }
                    else if (measure_name == F1Score::MICRO_NAME) {
                        measure = make_shared<F1Score>(
                            pgm_estimator, threshold, frequency_type, false);
                    }
                    else if (measure_name == RMSE::NAME) {
                        measure =
                            make_shared<RMSE>(pgm_estimator, frequency_type);
                    }

                    measure->set_fixed_steps(fixed_time_steps);
                    this->evaluation->add_measure(measure);
                }
            }
            else {
                throw TomcatModelException(
                    "Currently, evaluations are only supported "
                    "for PGM estimators.");
            }
        }

        void Experimentation::evaluate_and_save(const string& params_dir,
                                                int num_folds,
                                                const string& eval_dir,
                                                const EvidenceSet& data,
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
                if (train_dir.empty()) {
                    data_splitter = make_shared<DataSplitter>(empty_set, data);
                }
                else {
                    // The baseline method outputs probabilities as the
                    // frequencies in the values of the samples used for
                    // training. If this executable is called with baseline
                    // set, we assume that the content of the parameter data
                    // is the training data.
                    EvidenceSet train_data(train_dir);
                    data_splitter = make_shared<DataSplitter>(train_data, data);
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

            if (!params_dir.empty()) {
                shared_ptr<ModelTrainer> loader;
                if (const auto& dbn =
                        dynamic_pointer_cast<DynamicBayesNet>(this->model)) {
                    loader =
                        make_shared<DBNLoader>(dbn, final_params_dir, true);
                }
                else {
                    loader = make_shared<ModelLoader>(
                        this->model, final_params_dir, true);
                }
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
            const std::string& params_dir, const string& eval_dir) {
            // Data comes from the message bus
            EvidenceSet empty_set;
            shared_ptr<DataSplitter> data_splitter =
                make_shared<DataSplitter>(empty_set, empty_set);

            this->estimation->get_agent()->show_progress(false);

            unique_ptr<Pipeline> pipeline;
            if (eval_dir.empty()) {
                pipeline = make_unique<Pipeline>();
            }
            else {
                fs::create_directories(eval_dir);
                string filepath =
                    fmt::format("{}/{}.json", eval_dir, this->experiment_id);
                ofstream output_file;
                output_file.open(filepath);

                this->estimation->set_display_estimates(true);
                pipeline =
                    make_unique<Pipeline>(this->experiment_id, output_file);
            }

            pipeline->set_data_splitter(data_splitter);
            if (!params_dir.empty()) {
                shared_ptr<ModelTrainer> loader;
                if (const auto& dbn =
                        dynamic_pointer_cast<DynamicBayesNet>(this->model)) {
                    loader = make_shared<DBNLoader>(dbn, params_dir, true);
                }
                else {
                    loader =
                        make_shared<ModelLoader>(this->model, params_dir, true);
                }
                pipeline->set_model_trainer(loader);
            }
            pipeline->set_estimation_process(this->estimation);
            pipeline->execute();
        }

        void Experimentation::generate_synthetic_data(
            const string& params_dir,
            const string& data_dir,
            int num_data_samples,
            int num_time_steps,
            int equal_samples_time_step_limit,
            const unordered_set<string>& exclusions,
            int num_jobs) {

            if (const auto& dbn =
                    dynamic_pointer_cast<DynamicBayesNet>(this->model)) {

                dbn->unroll(num_time_steps, true);

                if (!params_dir.empty()) {
                    shared_ptr<ModelTrainer> loader =
                        make_shared<DBNLoader>(dbn, params_dir, true);
                    loader->fit(EvidenceSet());
                }

                AncestralSampler sampler(dbn, num_jobs);
                sampler.set_num_in_plate_samples(num_data_samples);
                sampler.set_equal_samples_time_step_limit(
                    equal_samples_time_step_limit);
                sampler.sample(this->random_generator, num_data_samples);
                sampler.save_samples_to_folder(data_dir, exclusions);
            }
            else {
                throw TomcatModelException(
                    "Data can only be generated from a DBN model.");
            }
        }

        void Experimentation::print_model(const std::string& params_dir,
                                          const std::string& model_dir) {

            if (const auto& dbn =
                    dynamic_pointer_cast<DynamicBayesNet>(this->model)) {

                dbn->unroll(3, true);
                shared_ptr<ModelTrainer> loader =
                    make_shared<DBNLoader>(dbn, params_dir, true);
                loader->fit(EvidenceSet());

                fs::create_directories(model_dir);
                ofstream output_file;
                string graph_filepath = get_filepath(model_dir, "graph.viz");
                output_file.open(graph_filepath);
                dbn->print_graph(output_file);
                output_file.close();

                if (!params_dir.empty()) {
                    string cpds_filepath = get_filepath(model_dir, "cpds.txt");
                    output_file.open(cpds_filepath);
                    dbn->print_cpds(output_file);
                    output_file.close();
                }
            }
            else {
                throw TomcatModelException(
                    "Currently, only DBN models can be printed.");
            }
        }

        //----------------------------------------------------------------------
        // Getters & Setters
        //----------------------------------------------------------------------

        void Experimentation::set_model(const shared_ptr<Model>& new_model) {
            Experimentation::model = new_model;
        }

    } // namespace model
} // namespace tomcat
