#include "Experimentation.h"

#include <boost/filesystem.hpp>
#include <nlohmann/json.hpp>

#include "pipeline/estimation/CompoundSamplerEstimator.h"
#include "pipeline/estimation/SamplerEstimator.h"
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
            const std::string& experiment_id,
            std::shared_ptr<DynamicBayesNet>& model)
            : random_generator(gen), experiment_id(experiment_id),
              model(model) {

            this->offline_estimation = make_shared<OfflineEstimation>();
            // Include the estimates updated at every time step in the final
            // evaluation document.
            this->offline_estimation->set_display_estimates(true);
        }

        Experimentation::Experimentation(const shared_ptr<gsl_rng>& gen,
                                         const string& experiment_id,
                                         MODEL_VERSION model_version,
                                         const EvidenceSet& training_set,
                                         const EvidenceSet& test_set)
            : random_generator(gen), experiment_id(experiment_id) {

            this->init_model(model_version);
            this->data_splitter =
                make_shared<DataSplitter>(training_set, test_set);
            this->offline_estimation = make_shared<OfflineEstimation>();
        }

        Experimentation::Experimentation(const shared_ptr<gsl_rng>& gen,
                                         const string& experiment_id,
                                         MODEL_VERSION model_version,
                                         const EvidenceSet& data,
                                         int num_folds)
            : random_generator(gen), experiment_id(experiment_id) {

            this->init_model(model_version);
            this->data_splitter =
                make_shared<DataSplitter>(data, num_folds, gen);
            this->offline_estimation = make_shared<OfflineEstimation>();
        }

        Experimentation::Experimentation(const shared_ptr<gsl_rng>& gen,
                                         MODEL_VERSION model_version)
            : random_generator(gen) {
            this->init_model(model_version);
        }

        Experimentation::~Experimentation() {}

        //----------------------------------------------------------------------
        // Operator overload
        //----------------------------------------------------------------------

        //----------------------------------------------------------------------
        // Static functions
        //----------------------------------------------------------------------

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
                splitter = DataSplitter(
                    data, num_folds, this->random_generator);
                final_params_dir = fmt::format("{}/fold{{}}", params_dir);
                splitter.save_indices(params_dir);
            }

            DBNSaver model_saver(
                this->model, this->trainer, final_params_dir, true);

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
                                                       int num_jobs) {
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
                if (!this->model->is_exact_inference_allowed()) {
                    shared_ptr<GibbsSampler> gibbs_sampler =
                        make_shared<GibbsSampler>(
                            this->model, burn_in, num_jobs);
                    approximate_estimator =
                        make_shared<CompoundSamplerEstimator>(
                            move(gibbs_sampler),
                            this->random_generator,
                            num_samples);
                    this->offline_estimation->add_estimator(
                        approximate_estimator);
                }

                for (const auto& inference_item : json_inference) {
                    Eigen::VectorXd value(0);
                    if (inference_item["value"] != "") {
                        value = Eigen::VectorXd::Constant(
                            1, stod((string)inference_item["value"]));
                    }

                    const string& variable_label = inference_item["variable"];

                    shared_ptr<Estimator> model_estimator;
                    if (this->model->is_exact_inference_allowed()) {
                        model_estimator = make_shared<SumProductEstimator>(
                            this->model,
                            inference_item["horizon"],
                            variable_label,
                            value);

                        this->offline_estimation->add_estimator(
                            model_estimator);
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

                    // Add baseline estimator
                    //                    shared_ptr<Estimator>
                    //                    baseline_estimator =
                    //                        make_shared<TrainingFrequencyEstimator>(
                    //                            this->tomcat->get_model(),
                    //                            inference_item["horizon"],
                    //                            variable_label,
                    //                            value);
                    //                    this->offline_estimation->add_estimator(baseline_estimator);

                    // Evaluation metrics
                    bool eval_last_only =
                        this->model->get_nodes_by_label(variable_label)
                            .size() == 1;
                    this->evaluation->add_measure(make_shared<Accuracy>(
                        model_estimator, 0.5, eval_last_only));
                    //                    this->evaluation->add_measure(make_shared<Accuracy>(
                    //                        baseline_estimator, 0.5,
                    //                        eval_last_only));
                    if (value.size() > 0) {
                        this->evaluation->add_measure(
                            make_shared<F1Score>(model_estimator, 0.5));
                        //                        this->evaluation->add_measure(
                        //                            make_shared<F1Score>(baseline_estimator,
                        //                            0.5));
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
                                                const EvidenceSet& data) {
            fs::create_directories(eval_dir);
            string filepath =
                fmt::format("{}/{}.json", eval_dir, this->experiment_id);
            ofstream output_file;
            output_file.open(filepath);

            shared_ptr<DataSplitter> data_splitter;
            string final_params_dir;
            if (num_folds > 1) {
                // One set of learned parameters per fold
                final_params_dir = fmt::format("{}/fold{{}}", params_dir);
                data_splitter = make_shared<DataSplitter>(data, params_dir);
            }
            else {
                final_params_dir = params_dir;
                EvidenceSet empty_training_data;
                data_splitter =
                    make_shared<DataSplitter>(empty_training_data, data);
            }

            shared_ptr<DBNTrainer> loader =
                make_shared<DBNLoader>(this->model, final_params_dir, true);
            EvidenceSet empty_training;

            Pipeline pipeline(this->experiment_id, output_file);
            pipeline.set_data_splitter(data_splitter);
            pipeline.set_model_trainer(loader);
            pipeline.set_estimation_process(this->offline_estimation);
            pipeline.set_aggregator(this->evaluation);
            pipeline.execute();
            output_file.close();
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

        bool Experimentation::should_eval_last_only(const string& node_label) {
            // If the node is not repeatable in the unrolled DBN, we evaluate
            // the accuracy using the estimate in the last time step which will
            // give us the distribution of the node given all the observations
            // in the mission trial.
            return this->tomcat->get_model()
                       ->get_nodes_by_label(node_label)
                       .size() == 1;
        }

        void Experimentation::init_model(MODEL_VERSION model_version) {
            switch (model_version) {
            case v1: {
                this->tomcat = make_shared<TomcatTA3>();
                this->data_generation_exclusions = {TomcatTA3::STATE};
                break;
            }
            case v2: {
                this->tomcat = make_shared<TomcatTA3V2>();
                this->data_generation_exclusions = {TomcatTA3V2::STATE,
                                                    TomcatTA3V2::PBAE};
                break;
            }
            }
            this->tomcat->init();
        }

        void Experimentation::display_estimates() {
            this->offline_estimation->set_display_estimates(true);
        }

        void Experimentation::load_model_from(const string& input_dir) {
            string final_input_dir = input_dir;

            if (this->data_splitter &&
                this->data_splitter->get_splits().size() > 1) {
                final_input_dir = fmt::format("{}/fold{{}}", input_dir);
            }

            this->trainer = make_shared<DBNLoader>(this->tomcat->get_model(),
                                                   final_input_dir);
        }

        void Experimentation::train_using_gibbs(int burn_in,
                                                int num_samples,
                                                int num_jobs) {
            this->trainer = make_shared<DBNSamplingTrainer>(
                this->random_generator,
                make_shared<GibbsSampler>(
                    this->tomcat->get_model(), burn_in, num_jobs),
                num_samples);
        }

        void Experimentation::save_model(const string& output_dir,
                                         bool save_partials) {
            string final_output_dir = output_dir;

            if (this->data_splitter->get_splits().size() > 1) {
                final_output_dir = fmt::format("{}/fold{{}}", final_output_dir);
            }

            this->saver = make_shared<DBNSaver>(this->tomcat->get_model(),
                                                this->trainer,
                                                final_output_dir,
                                                save_partials);
        }

        void Experimentation::compute_baseline_estimates_for(
            const string& node_label,
            int inference_horizon,
            const Eigen::VectorXd& assignment) {

            this->display_estimates();
            shared_ptr<Estimator> estimator =
                make_shared<TrainingFrequencyEstimator>(
                    this->tomcat->get_model(),
                    inference_horizon,
                    node_label,
                    assignment);
            this->offline_estimation->add_estimator(estimator);
        }

        void Experimentation::compute_estimates_for(
            const string& node_label,
            int inference_horizon,
            const Eigen::VectorXd& assignment) {

            this->display_estimates();
            shared_ptr<Estimator> estimator =
                make_shared<SumProductEstimator>(this->tomcat->get_model(),
                                                 inference_horizon,
                                                 node_label,
                                                 assignment);
            this->offline_estimation->add_estimator(estimator);
        }

        void Experimentation::compute_baseline_eval_scores_for(
            const string& node_label,
            int inference_horizon,
            const vector<MEASURES>& measures,
            const Eigen::VectorXd& assignment) {

            shared_ptr<Estimator> estimator =
                make_shared<TrainingFrequencyEstimator>(
                    this->tomcat->get_model(),
                    inference_horizon,
                    node_label,
                    assignment);

            this->compute_eval_scores_for(
                node_label, inference_horizon, measures, assignment, estimator);
        }

        void Experimentation::compute_eval_scores_for(
            const string& node_label,
            int inference_horizon,
            const vector<MEASURES>& measures,
            const Eigen::VectorXd& assignment,
            const shared_ptr<Estimator>& estimator) {

            if (!measures.empty()) {
                this->init_evaluation();
                bool eval_last_only = this->should_eval_last_only(node_label);

                this->offline_estimation->add_estimator(estimator);

                for (const auto& measure : measures) {
                    if (measure == MEASURES::accuracy) {
                        this->evaluation->add_measure(make_shared<Accuracy>(
                            estimator, 0.5, eval_last_only));
                    }
                    else if (measure == MEASURES::f1) {
                        this->evaluation->add_measure(
                            make_shared<F1Score>(estimator, 0.5));
                    }
                }
            }
        }

        void Experimentation::init_evaluation() {
            if (!this->evaluation) {
                this->evaluation = make_shared<EvaluationAggregator>(
                    EvaluationAggregator::METHOD::no_aggregation);
            }
        }

        void Experimentation::compute_eval_scores_for(
            const string& node_label,
            int inference_horizon,
            const vector<MEASURES>& measures,
            const Eigen::VectorXd& assignment) {

            shared_ptr<Estimator> estimator =
                make_shared<SumProductEstimator>(this->tomcat->get_model(),
                                                 inference_horizon,
                                                 node_label,
                                                 assignment);

            this->compute_eval_scores_for(
                node_label, inference_horizon, measures, assignment, estimator);
        }

        void Experimentation::train_and_evaluate(const string& output_dir,
                                                 bool evaluate_on_partials) {
            fs::create_directories(output_dir);
            string filepath = fmt::format("{}/evaluations.json", output_dir);
            ofstream output_file;
            output_file.open(filepath);
            Pipeline pipeline(this->experiment_id, output_file);
            pipeline.set_data_splitter(this->data_splitter);
            pipeline.set_model_trainer(this->trainer);
            pipeline.set_model_saver(this->saver);
            pipeline.set_estimation_process(this->offline_estimation);
            pipeline.set_aggregator(this->evaluation);
            pipeline.execute();
            output_file.close();

            if (evaluate_on_partials) {
                this->evaluate_on_partials(output_dir);
                // Restore the model's parameters to the aggregation over
                // partials.
                int last_split_idx =
                    this->data_splitter->get_splits().size() - 1;
                this->trainer->update_model_from_partials(last_split_idx, true);
            }
        }

        void Experimentation::evaluate_on_partials(const string& output_dir) {
            cout << "\nComputing estimates over partials\n";

            string partials_dir = fmt::format("{}/partials", output_dir);
            if (this->data_splitter->get_splits().size() > 1) {
                partials_dir = fmt::format("{}/fold{{}}", partials_dir);
            }

            int split_idx = 0;
            for (const auto& [training_data, test_data] :
                 this->data_splitter->get_splits()) {
                cout << "\nFold " << (split_idx + 1) << "\n";

                for (int i = 0; i < this->trainer->get_num_partials(); i++) {
                    cout << "\nPartial " << (i + 1) << "\n";

                    const string filename =
                        fmt::format("evaluations{}.json", i + 1);
                    const string experiment_id =
                        fmt::format("{}_{}", this->experiment_id, i + 1);

                    // Since the model is a pointer, updating the parameters
                    // here will affect the estimates over this model.
                    this->trainer->update_model_from_partial(
                        i, split_idx, true);

                    const string eval_dir =
                        fmt::format(partials_dir, split_idx + 1);
                    const string filepath = get_filepath(eval_dir, filename);
                    fs::create_directories(eval_dir);

                    shared_ptr<DataSplitter> data_splitter =
                        make_shared<DataSplitter>(training_data, test_data);

                    ofstream output_file;
                    output_file.open(filepath);
                    Pipeline pipeline(experiment_id, output_file);
                    pipeline.set_data_splitter(data_splitter);
                    pipeline.set_estimation_process(this->offline_estimation);
                    pipeline.set_aggregator(this->evaluation);
                    pipeline.execute();
                    output_file.close();
                }

                split_idx++;
            }
        }

        void Experimentation::train_and_save() {
            DataSplitter splitter(this->training_data, this->test_data);

            for (const auto& [training_data, test_data] :
                 splitter.get_splits()) {
                this->trainer->prepare();
                this->trainer->fit(training_data);
                this->saver->save();
            }
        }

        void
        Experimentation::generate_synthetic_data(int num_samples,
                                                 const std::string& output_dir,
                                                 int equals_until,
                                                 int max_time_step) {

            this->trainer->fit({});
            this->tomcat->generate_synthetic_data(
                this->random_generator,
                num_samples,
                output_dir,
                equals_until,
                max_time_step,
                this->data_generation_exclusions);
        }

        void
        Experimentation::set_training_data(const EvidenceSet& training_data) {
            this->training_data = training_data;
        }

        void Experimentation::set_test_data(const EvidenceSet& test_data) {
            this->test_data = test_data;
        }

        //----------------------------------------------------------------------
        // Getters & Setters
        //----------------------------------------------------------------------

    } // namespace model
} // namespace tomcat
