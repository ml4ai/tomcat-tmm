#pragma once

#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include <gsl/gsl_rng.h>

#include "converter/MessageConverter.h"
#include "pgm/DynamicBayesNet.h"
#include "pgm/EvidenceSet.h"
#include "pipeline/estimation/Agent.h"
#include "pipeline/estimation/EstimationProcess.h"
#include "pipeline/evaluation/EvaluationAggregator.h"
#include "pipeline/training/DBNTrainer.h"

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
             * Initializes an experiment for offline estimation.
             *
             * @param gen: random number generator
             * @param experiment_id: id of the experiment
             * @param model: model to experiment on
             * @param real_time_estimation: whether estimates are computed on
             * the fly
             */
            Experimentation(const std::shared_ptr<gsl_rng>& gen,
                            const std::string& experiment_id,
                            const std::shared_ptr<DynamicBayesNet>& model);

            /**
             * Initializes an agent for online estimation.
             *
             * @param gen: random number generator
             * @param model: model to compute estimates from
             * @param agent: agent who talks to the message bus
             * @param broker_address: address of the message broker
             * @param broker_port: port of the message broker
             * @param num_connection_trials: number of attempts to connect
             * with the message broker
             * @param milliseconds_before_retrial: milliseconds to wait
             * before retrying to connect with the message broker in case of
             * fail to connect previously
             */
            Experimentation(const std::shared_ptr<gsl_rng>& gen,
                            const std::shared_ptr<DynamicBayesNet>& model,
                            const std::shared_ptr<Agent>& agent,
                            const std::string& broker_address,
                            int broker_port,
                            int num_connection_trials,
                            int milliseconds_before_retrial);

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
             * Adds a series of estimators to evaluate inferences in a
             * pre-trained model.
             *
             * @param filepath: filepath to a json file containing the
             * specifications of each estimation needed
             * @param burn_in: burn in period if approximate inference is
             * necessary
             * @param num_samples: number of samples after burn in
             * @param num_jobs: number of jobs used to perform the computations
             * @param baseline: whether to use the baseline estimator based
             * on frequencies of values of training samples
             */
            void add_estimators_from_json(const std::string& filepath,
                                          int burn_in,
                                          int num_samples,
                                          int num_jobs,
                                          bool baseline);

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
            void evaluate_and_save(const std::string& params_dir,
                                   int num_folds,
                                   const std::string& eval_dir,
                                   const EvidenceSet& data,
                                   bool baseline);

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

          private:
            //------------------------------------------------------------------
            // Data members
            //------------------------------------------------------------------
            std::shared_ptr<gsl_rng> random_generator;

            std::shared_ptr<DynamicBayesNet> model;

            std::shared_ptr<DBNTrainer> trainer;

            std::shared_ptr<EstimationProcess> estimation;

            std::shared_ptr<EvaluationAggregator> evaluation;

            std::string experiment_id;
        };

    } // namespace model
} // namespace tomcat
