#pragma once

#include <nlohmann/json.hpp>

#include "pgm/DynamicBayesNet.h"
#include "pgm/EvidenceSet.h"
#include "utils/Definitions.h"
#include "utils/Tensor3.h"

namespace tomcat {
    namespace model {

        /**
         * Class responsible for estimating a model's parameters.
         */
        class DBNTrainer {
          public:
            //------------------------------------------------------------------
            // Constructors & Destructor
            //------------------------------------------------------------------

            /**
             * Creates an abstract trainer.
             */
            DBNTrainer();

            virtual ~DBNTrainer();

            //------------------------------------------------------------------
            // Copy & Move constructors/assignments
            //------------------------------------------------------------------

            // Copy constructor and assignment should be deleted to avoid
            // implicit slicing and loss of polymorphic behaviour in the
            // subclasses.
            DBNTrainer(const DBNTrainer&) = delete;

            DBNTrainer& operator=(const DBNTrainer&) = delete;

            DBNTrainer(DBNTrainer&&) = default;

            DBNTrainer& operator=(DBNTrainer&&) = default;

            //------------------------------------------------------------------
            // Virtual functions
            //------------------------------------------------------------------

            /**
             * Prepares the trainer to a series of calls to the function fit by
             * performing necessary cleanups.
             */
            virtual void prepare();

            //------------------------------------------------------------------
            // Pure virtual functions
            //------------------------------------------------------------------

            /**
             * Estimates the model's parameters from training data. The final
             * parameters are defined as the average over the samples
             * (partials).
             */
            virtual void fit(const EvidenceSet& training_data) = 0;

            /**
             * Writes information about the splitter in a json object.
             *
             * @param json: json object
             */
            virtual void get_info(nlohmann::json& json) const = 0;

            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            /**
             * Returns the samples generated for each parameter in the model
             * during the training process.
             *
             * @param split_idx: Index of the data split.
             *
             * @return Parameter samples.
             */
            std::unordered_map<std::string, Tensor3>
            get_partials(int split_idx) const;

            /**
             * Returns the number of parameter samples stored in the trainer.
             *
             * @return Number of parameter samples.
             */
            int get_num_partials() const;

            /**
             * Updates model using parameters from a given sample generated
             * during the training process.
             *
             * @param sample_idx: Index of the sample.
             * @param split_idx: Index of the data split.
             * @param force: whether frozen parameter nodes should be updated.
             */
            void update_model_from_partial(int sample_idx,
                                           int split_idx,
                                           bool force);

            /**
             * Updates model using as parameters the average over the
             * parameter samples generated in the trained step.
             *
             * @param split_idx: Index of the data split.
             *
             * @param force: whether frozen parameter nodes should be updated.
             */
            void update_model_from_partials(int split_idx, bool force);

          protected:
            //------------------------------------------------------------------
            // Data members
            //------------------------------------------------------------------

            // Vector of mapping between a parameter node's label and the
            // list of samples generated from it. The vector store this
            // information for each data split.
            std::vector<std::unordered_map<std::string, Tensor3>>
                param_label_to_samples;

            //------------------------------------------------------------------
            // Pure virtual functions
            //------------------------------------------------------------------

            /**
             * Gets the model from a specific instance of a trainer.
             */
            virtual std::shared_ptr<DynamicBayesNet> get_model() const = 0;

          private:
            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            /**
             * Updates a model either by using a specific sample from the
             * partials or by using the average over all partials.
             *
             * @param sample_idx: index of the samples to use. If nullptr, an
             * average over the partials are used as parameter value.
             * @param split_idx: Index of the data split.
             * @param force: whether frozen parameter nodes should be updated.
             */
            void update_model(std::unique_ptr<int> sample_idx,
                              int split_idx,
                              bool force);
        };

    } // namespace model
} // namespace tomcat
