#pragma once

#include <nlohmann/json.hpp>

#include "pgm/DynamicBayesNet.h"
#include "pgm/EvidenceSet.h"
#include "utils/Definitions.h"
#include "utils/Tensor3.h"
#include "pipeline/training/ModelTrainer.h"

namespace tomcat {
    namespace model {

        /**
         * Class responsible for estimating a model's parameters.
         */
        class DBNTrainer : public ModelTrainer {
          public:
            //------------------------------------------------------------------
            // Constructors & Destructor
            //------------------------------------------------------------------

            /**
             * Creates an abstract trainer.
             */
            explicit DBNTrainer(const DBNPtr& model);

            virtual ~DBNTrainer() = default;

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
            // Member functions
            //------------------------------------------------------------------

            void prepare() override;

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
            void update_model(const std::unique_ptr<int>& sample_idx,
                              int split_idx,
                              bool force);
        };

    } // namespace model
} // namespace tomcat
