#include "Experimentation.h"

#include <boost/filesystem.hpp>
#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include "pipeline/DBNSaver.h"
#include "pipeline/DataSplitter.h"
#include "pipeline/Pipeline.h"
#include "pipeline/estimation/CompoundSamplerEstimator.h"
#include "pipeline/estimation/OfflineEstimation.h"
#include "pipeline/estimation/OnlineEstimation.h"
#include "pipeline/estimation/SamplerEstimator.h"
#include "pipeline/estimation/SumProductEstimator.h"
#include "pipeline/estimation/TrainingFrequencyEstimator.h"
#include "pipeline/evaluation/Accuracy.h"
#include "pipeline/evaluation/F1Score.h"
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
              model(model) {

            this->estimation = make_shared<OfflineEstimation>();
            // Include the estimates updated at every time step in the final
            // evaluation document.
            this->estimation->set_display_estimates(true);
        }

        Experimentation::Experimentation(
            const shared_ptr<gsl_rng>& gen,
            const shared_ptr<DynamicBayesNet>& model,
            const std::shared_ptr<Agent>& agent,
            const string& broker_address,
            int broker_port,
            int num_connection_trials,
            int milliseconds_before_retrial)
            : random_generator(gen), model(model) {

            OnlineEstimation::MessageBrokerConfiguration config;
            config.timeout = INT32_MAX;
            config.address = broker_address;
            config.port = broker_port;
            config.num_connection_trials = num_connection_trials;
            config.milliseconds_before_retrial = milliseconds_before_retrial;
            this->estimation = make_shared<OnlineEstimation>(config, agent);
        }

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

        void Experimentation::add_estimators_from_json(const string& filepath,
                                                       int burn_in,
                                                       int num_samples,
                                                       int num_jobs,
                                                       bool baseline,
                                                       bool exact_inference) {
            fstream file;
            file.open(filepath);
            if (file.is_open()) {
                nlohmann::json json_inference = nlohmann::json::parse(file);

                if (json_inference.empty()) {
                    stringstream ss;
                    ss << "Nothing to Infer. The file " << filepath
                       << "is empty.";
                    throw TomcatModelException(ss.str());
                }

                this->evaluation = make_shared<EvaluationAggregator>(
                    EvaluationAggregator::METHOD::no_aggregation);

                shared_ptr<CompoundSamplerEstimator> approximate_estimator;
                if (!exact_inference && !baseline) {
                    shared_ptr<GibbsSampler> gibbs_sampler =
                        make_shared<GibbsSampler>(
                            this->model, burn_in, num_jobs);
                    approximate_estimator =
                        make_shared<CompoundSamplerEstimator>(
                            move(gibbs_sampler),
                            this->random_generator,
                            num_samples);
                    this->estimation->add_estimator(approximate_estimator);
                }

                for (const auto& inference_item : json_inference) {
                    Eigen::VectorXd value(0);
                    if (inference_item["value"] != "") {
                        value = Eigen::VectorXd::Constant(
                            1, stod((string)inference_item["value"]));
                    }

                    const string& variable_label = inference_item["variable"];

                    shared_ptr<Estimator> model_estimator;
                    if (baseline) {
                        model_estimator =
                            make_shared<TrainingFrequencyEstimator>(
                                this->model,
                                inference_item["horizon"],
                                variable_label,
                                value);
                        this->estimation->add_estimator(model_estimator);
                    }
                    else {
                        if (exact_inference) {
                            model_estimator = make_shared<SumProductEstimator>(
                                this->model,
                                inference_item["horizon"],
                                variable_label,
                                value);

                            this->estimation->add_estimator(model_estimator);
                        }
                        else {
                            shared_ptr<SamplerEstimator> sampler_estimator =
                                make_shared<SamplerEstimator>(
                                    this->model,
                                    inference_item["horizon"],
                                    variable_label,
                                    value);
                            approximate_estimator->add_base_estimator(
                                sampler_estimator);
                            model_estimator = sampler_estimator;
                        }
                    }

                    // Evaluation metrics
                    bool eval_last_only =
                        this->model->get_nodes_by_label(variable_label)
                            .size() == 1;
                    this->evaluation->add_measure(make_shared<Accuracy>(
                        model_estimator, 0.5, eval_last_only));
                    if (value.size() > 0) {
                        this->evaluation->add_measure(
                            make_shared<F1Score>(model_estimator, 0.5));
                    }
                }
            }
            else {
                stringstream ss;
                ss << "The file " << filepath << " does not exist.";
                throw TomcatModelException(ss.str());
            }
        }

        void Experimentation::evaluate_and_save(const string& params_dir,
                                                int num_folds,
                                                const string& eval_dir,
                                                const EvidenceSet& data,
                                                bool baseline,
                                                const string& train_dir,
                                                bool only_estimates) {
//            FactorGraph::create_from_unrolled_dbn(*this->model).print_graph(cout);
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

            Pipeline pipeline(this->experiment_id, output_file);
            pipeline.set_data_splitter(data_splitter);
            pipeline.set_model_trainer(loader);
            pipeline.set_estimation_process(this->estimation);
            if (!only_estimates) {
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

            for (auto& estimator : this->estimation->get_estimators()) {
                estimator->set_show_progress(false);
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
