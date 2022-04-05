#pragma once

#include <memory>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

#include <eigen3/Eigen/Dense>
#include <nlohmann/json.hpp>

#include "pgm/EvidenceSet.h"
#include "pipeline/Model.h"
#include "utils/Definitions.h"
#include "utils/Tensor3.h"
#include "pipeline/estimation/OnlineLogger.h"

namespace tomcat {
    namespace model {

        /**
         * Represents a generic estimator for a DBN model.
         */
        class Estimator {
          public:
            //------------------------------------------------------------------
            // Types, Enums & Constants
            //------------------------------------------------------------------

            enum FREQUENCY_TYPE { all, last, fixed, dynamic };

            //------------------------------------------------------------------
            // Constructors & Destructor
            //------------------------------------------------------------------

            Estimator() = default;

            explicit Estimator(const std::shared_ptr<Model>& model);

            virtual ~Estimator() = default;

            //------------------------------------------------------------------
            // Copy & Move constructors/assignments
            //------------------------------------------------------------------

            // Copy constructor and assignment should be deleted to avoid
            // implicit slicing and loss of polymorphic behaviour in the
            // subclasses. To deep copy, the clone method must be used.
            Estimator(const Estimator&) = delete;

            Estimator& operator=(const Estimator&) = delete;

            Estimator(Estimator&&) = default;

            Estimator& operator=(Estimator&&) = default;

            //------------------------------------------------------------------
            // Static functions
            //------------------------------------------------------------------

            /**
             * Creates an instance of a custom estimator.
             *
             * @param estimator_name: name of the estimator
             * @param model: underlying model
             * @param json_settings: settings in a json object
             * @param frequency_type: frequency type for inferences
             * @param fixed_time_steps: fixed time steps when to compute
             * inferences
             * @param num_jobs: number of jobs to use for parallel processing
             *
             * @return Custom estimator
             */
            static EstimatorPtr
            factory(const std::string& estimator_name,
                    const ModelPtr& model,
                    const nlohmann::json& json_settings,
                    FREQUENCY_TYPE frequency_type,
                    const std::unordered_set<int>& fixed_time_steps,
                    int num_jobs);

            //------------------------------------------------------------------
            // Virtual functions
            //------------------------------------------------------------------

            /**
             * Clean up at the end of a pipeline.
             */
            virtual void cleanup();

            /**
             * Prepare for another round of executions within a single pipeline.
             */
            virtual void prepare();

            /**
             * Store the last estimates computed in the pipeline.
             */
            virtual void keep_estimates();

            /**
             * Whether the estimator should print its progress on screen.
             *
             * @param show_progress: true or false
             */
            virtual void set_show_progress(bool show_progress);

            /**
             * Sets a logger for online estimation
             *
             * @param logger: logger
             */
            virtual void set_logger(const OnlineLoggerPtr& logger);

            //------------------------------------------------------------------
            // Pure virtual functions
            //------------------------------------------------------------------

            /**
             * Computes new estimates for the new data. New data consists of
             * observed values for time steps after the last processed one.
             *
             * @param new_data: observed values for time steps not already
             * seen by the estimator
             */
            virtual void estimate(const EvidenceSet& new_data) = 0;

            /**
             * Writes information about the estimator in a json object.
             *
             * @param json: json object
             */
            virtual void get_info(nlohmann::json& json) const = 0;

            /**
             * Returns the name of the estimator.
             *
             * @return Estimator's name.
             */
            virtual std::string get_name() const = 0;

            //------------------------------------------------------------------
            // Getters & Setters
            //------------------------------------------------------------------
            void set_training_data(const EvidenceSet& training_data);

            const std::shared_ptr<Model>& get_model() const;

          protected:
            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            /**
             * Copies data members from another estimator.
             */
            void copy(const Estimator& estimator);

            //------------------------------------------------------------------
            // Data members
            //------------------------------------------------------------------
            std::shared_ptr<Model> model;

            // Data used to train the model. Baseline methods can use the
            // information in the training set to compute their estimations
            // instead of test data.
            EvidenceSet training_data;

            // Whether a progress bar must be shown as the estimations are
            // happening
            bool show_progress = true;

            OnlineLoggerPtr logger;
        };

    } // namespace model
} // namespace tomcat
