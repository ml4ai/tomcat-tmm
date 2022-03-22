#pragma once

#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include <gsl/gsl_rng.h>

#include "pgm/DynamicBayesNet.h"
#include "pgm/EvidenceSet.h"
#include "pipeline/DBNSaver.h"
#include "pipeline/ModelSaver.h"
#include "pipeline/estimation/Agent.h"
#include "pipeline/estimation/EstimationProcess.h"
#include "pipeline/estimation/Estimator.h"
#include "pipeline/evaluation/EvaluationAggregator.h"
#include "pipeline/training/DBNTrainer.h"
#include "pipeline/training/ModelTrainer.h"

namespace tomcat {
    namespace model {

        /**
         * Class description here
         */
        class Experimentation {
          public:
            //------------------------------------------------------------------
            // Constructors & Destructor
            //------------------------------------------------------------------

            /**
             * Initializes an experiment for offline estimation without an
             * estimate report.
             *
             * @param gen: random number generator
             * @param experiment_id: id of the experiment
             */
            Experimentation(const std::shared_ptr<gsl_rng>& gen,
                            const std::string& experiment_id);

            ~Experimentation();

            //------------------------------------------------------------------
            // Copy & Move constructors/assignments
            //------------------------------------------------------------------
            Experimentation(const Experimentation&) = default;

            Experimentation& operator=(const Experimentation&) = default;

            Experimentation(Experimentation&&) = default;

            Experimentation& operator=(Experimentation&&) = default;

            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            /**
             * Configure a Gibbs Sampler to train a model and estimate its
             * parameters.
             *
             * @param burn_in: burn in period
             * @param num_samples: number of samples after the burn in period
             * @param num_jobs: number of jobs used to perform the computations
             */
            void set_gibbs_trainer(int burn_in, int num_samples, int num_jobs);

            /**
             * Trains a model and save its parameters to files in a given
             * folder.
             *
             * @param params_dir: directory where learned parameters must be
             * saved
             * @param num_folds: number of folds (>1 for
             * experiments with cross-validation)
             * @param data: data used for training
             */
            void train_and_save(const std::string& params_dir,
                                int num_folds,
                                const EvidenceSet& data);

            /**
             * Configures an offline estimation process.
             *
             * @param agents_config_filepath: filepath to a json file containing
             * the specifications of each estimation needed
             * @param model_dir: directory where a json file with model
             * description is located
             * @param reporter_dir: directory where the reporter must be saved
             * (if a reporter was provided)
             * @param num_jobs: number of jobs used to perform the computations
             * @param max_time_step: maximum time step to project estimates in
             * case of variable horizon
             * @param estimate_reporter: produces json messages with estimates
             * in a specific format
             */
            void set_offline_estimation_process(
                const std::string& agent_config_filepath,
                const std::string& model_dir,
                const std::string& reporter_dir,
                int num_jobs,
                int max_time_step,
                const EstimateReporterPtr& estimate_reporter);

            /**
             * Configures an online estimation process.
             *
             * @param agents_config_filepath: filepath to a json file containing
             * the specifications of each estimation needed
             * @param model_dir: directory where a json file with model
             * description is located
             * @param num_jobs: number of jobs used to perform the computations
             * @param max_time_step: maximum time step to project estimates in
             * case of variable horizon
             * @param message_broker_config_filepath: filepath of the json file
             * containing details of the message broker for online estimation
             * @param converter: responsible for translating messages from the
             * message broker to model readable data
             * @param estimate_reporter: produces json messages with estimates
             * in a specific format
             */
            void set_online_estimation_process(
                const std::string& agent_config_filepath,
                const std::string& model_dir,
                int num_jobs,
                int max_time_step,
                const std::string& message_broker_config_filepath,
                const MsgConverterPtr& converter,
                const EstimateReporterPtr& estimate_reporter);

            /**
             * Evaluates a pre-trained model and save the evaluations to a
             * json file in a given directory.
             *
             * @param params_dir: directory where the parameters of a
             * pre-trained model are saved
             * @param num_folds: number of folds (>1 for experiments with
             * cross-validation)
             * @param eval_dir: directory where the final evaluation file
             * must be saved
             * @param data: test data (training data if the baseline
             * estimator is chosen), or full data fo experiments with
             * cross-validation
             * @param train_dir: directory where data used for training the
             * model is. This is only required for baseline evaluation.
             */
            void evaluate_and_save(const std::string& params_dir,
                                   int num_folds,
                                   const std::string& eval_dir,
                                   const EvidenceSet& data,
                                   const std::string& train_dir);

            /**
             * Evaluates a pre-trained model and save the evaluations to a
             * json file in a given directory.
             *
             * @param params_dir: directory where the parameters of a
             * pre-trained model are saved
             * @param num_folds: number of folds (>1 for experiments with
             * cross-validation)
             * @param eval_dir: directory where the final evaluation file
             * must be saved
             * @param data: test data (training data if the baseline
             * estimator is chosen), or full data fo experiments with
             * cross-validation
             * @param baseline: whether to use the baseline estimator based
             * on frequencies of values of training samples
             */
            void start_real_time_estimation(const std::string& params_dir);

            /**
             * Generates data samples from a pre-trained model.
             *
             * @param params_dir: directory where the parameters of a
             * pre-trained model are saved
             * @param data_dir: directory where the generated samples must be
             * saved
             * @param num_data_samples: number of data samples to generate
             * per variable of the model
             * @param num_time_steps: number of time steps to unroll the DBN
             * into
             * @param equal_samples_time_step_limit: time step up to when
             * samples must not differ (for each variable)
             * @param exclusions: json file containing a list of variable
             * labels for which samples must not be saved
             * @param num_jobs: number of jobs used to perform the computations
             */
            void generate_synthetic_data(
                const std::string& params_dir,
                const std::string& data_dir,
                int num_data_samples,
                int num_time_steps,
                int equal_samples_time_step_limit,
                const std::unordered_set<std::string>& exclusions,
                int num_jobs);

            /**
             * Prints the model structure and/or CPDs to files in a given
             * directory.
             *
             * @param params_dir: directory where the parameters of a
             * pre-trained model are saved
             * @param model_dir: directory where model's info must be saved
             */
            void print_model(const std::string& params_dir,
                             const std::string& model_dir);

            //------------------------------------------------------------------
            // Getters & Setters
            //------------------------------------------------------------------

            void set_model(const std::shared_ptr<Model>& model);

          private:
            //------------------------------------------------------------------
            // Static functions
            //------------------------------------------------------------------

            /**
             * Gets the frequency type specified in a json object.
             *
             * @param json_frequency: json object containing frequency
             * definition.
             *
             * @return Frequency type
             */
            static Estimator::FREQUENCY_TYPE
            get_frequency_type(const nlohmann::json& json_frequency);

            /**
             * Gets the fixed time steps specified in a json object.
             *
             * @param json_frequency: json object containing frequency
             * definition.
             *
             * @return Fixed time steps
             */
            static std::unordered_set<int>
            get_fixed_time_steps(const nlohmann::json& json_frequency);

            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            /**
             * Defines a saver class for the experiment.
             *
             * @param params_dir: directory where the model's parameters must be
             * saved
             * @param num_folds: number of folds to split the data
             */
            void set_model_saver(const std::string& params_dir, int num_folds);

            /**
             * Creates an agent from a json file containing its definitions.
             *
             * @param agent_config_filepath: filepath of the json file
             * @param model_dir: directory where a json file with model
             * description is located
             * @param num_jobs: number of jobs to use during estimation
             * @param max_time_step: maximum number of time steps to process
             * @return
             */
            AgentPtr create_agent(const std::string& agent_config_filepath,
                                  const std::string& model_dir,
                                  int num_jobs,
                                  int max_time_step);

            /**
             * Creates a model by parsing a json object containing some info
             * about how to create a model.
             *
             * @param json_model: json object
             * @param model_dir: directory where other model definition files
             * are located. It may be optional for some custom models.
             */
            void parse_model(const nlohmann::json& json_model,
                             const std::string& model_dir);

            /**
             * Parses a json object containing details of a list of
             * estimators and evaluations used and add the objects created
             * to an agent.
             *
             * @param json_estimators: json object
             * @param agent:agent
             * @param particle_filter_estimator: object responsible for
             * generating particles to be used by sampler estimators
             * @param num_jobs: number of jobs to use on estimation
             */
            void parse_estimators(
                const nlohmann::json& json_estimators,
                const AgentPtr& agent,
                const ParticleFilterEstimatorPtr& particle_filter_estimator,
                int num_jobs);

            /**
             * Parses a json object containing details of the evaluation to
             * be performed for a given estimator.
             *
             * @param json_evaluation: json object
             * @param estimator: estimator
             */
            void parse_evaluation(const nlohmann::json& json_evaluation,
                                  const EstimatorPtr& estimator);

            //------------------------------------------------------------------
            // Data members
            //------------------------------------------------------------------
            std::shared_ptr<gsl_rng> random_generator;

            std::shared_ptr<Model> model;

            std::shared_ptr<ModelTrainer> trainer;

            std::shared_ptr<ModelSaver> saver;

            std::shared_ptr<EstimationProcess> estimation;

            std::shared_ptr<EvaluationAggregator> evaluation;

            std::string experiment_id;
        };

    } // namespace model
} // namespace tomcat
