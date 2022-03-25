#pragma once

#include <memory>
#include <string>

#include "DBNTrainer.h"

#include "pipeline/Model.h"
#include "utils/Definitions.h"

namespace tomcat {
    namespace model {

        /**
         * Class responsible for loading a model from a pre-trained set of
         * parameters.
         */
        class ModelLoader : public ModelTrainer {
          public:
            //------------------------------------------------------------------
            // Constructors & Destructor
            //------------------------------------------------------------------

            /**
             * Creates an instance of the loader.
             *
             * @param model: model
             * @param input_folder_path: folder where the model's parameters
             * are stored
             * @param freeze_model: whether the model's parameter must be
             * frozen
             */
            ModelLoader(const ModelPtr& model,
                        const std::string& input_folder_path,
                        bool freeze_model = true);

            ~ModelLoader() = default;

            //------------------------------------------------------------------
            // Copy & Move constructors/assignments
            //------------------------------------------------------------------
            ModelLoader(const ModelLoader& loader);

            ModelLoader& operator=(const ModelLoader& loader);

            ModelLoader(ModelLoader&&) = default;

            ModelLoader& operator=(ModelLoader&&) = default;

            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------
            void prepare() override;

            void fit(const EvidenceSet& training_data) override;

            void get_info(nlohmann::json& json) const override;

          private:
            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            /**
             * Copies data members from another ModelLoader.
             */
            void copy(const ModelLoader& loader);

            //------------------------------------------------------------------
            // Data members
            //------------------------------------------------------------------

            // Folder where the model's parameters' files are saved.
            std::string input_folder_path;

            // Data split index. This is incremented at each call of the
            // function fit. It can be used to identify a folder with parameters
            // for a specific cross validation step.
            int split_idx = 0;

            // Freeze model
            bool freeze_model;
        };

    } // namespace model
} // namespace tomcat
