#pragma once

#include <nlohmann/json.hpp>

#include "pgm/EvidenceSet.h"
#include "utils/Definitions.h"
#include "pipeline/Model.h"

namespace tomcat {
    namespace model {

        /**
         * Class responsible for estimating a model's parameters.
         */
        class ModelTrainer {
          public:
            //------------------------------------------------------------------
            // Constructors & Destructor
            //------------------------------------------------------------------

            /**
             * Creates an abstract trainer.
             */
            explicit ModelTrainer(const ModelPtr& model) : model(model) {}

            virtual ~ModelTrainer() = default;

            //------------------------------------------------------------------
            // Copy & Move constructors/assignments
            //------------------------------------------------------------------

            // Copy constructor and assignment should be deleted to avoid
            // implicit slicing and loss of polymorphic behaviour in the
            // subclasses.
            ModelTrainer(const ModelTrainer&) = delete;

            ModelTrainer& operator=(const ModelTrainer&) = delete;

            ModelTrainer(ModelTrainer&&) = default;

            ModelTrainer& operator=(ModelTrainer&&) = default;

            //------------------------------------------------------------------
            // Virtual functions
            //------------------------------------------------------------------

            /**
             * Prepares the trainer to a series of calls to the function fit by
             * performing necessary cleanups.
             */
            virtual void prepare() = 0;

            //------------------------------------------------------------------
            // Pure virtual functions
            //------------------------------------------------------------------

            /**
             * Estimates the model's parameters from training data.
             */
            virtual void fit(const EvidenceSet& training_data) = 0;

            /**
             * Writes information about the trainer in a json object.
             *
             * @param json: json object
             */
            virtual void get_info(nlohmann::json& json) const = 0;

            //------------------------------------------------------------------
            // Getters & Setters
            //------------------------------------------------------------------

            const ModelPtr& get_model() const { return model; }

          protected:
            //------------------------------------------------------------------
            // Data members
            //-----------------------------------------------------------------

            ModelPtr model;
        };

    } // namespace model
} // namespace tomcat
