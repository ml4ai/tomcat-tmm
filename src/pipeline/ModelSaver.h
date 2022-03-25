#pragma once

#include <memory>
#include <string>

#include "utils/Definitions.h"
#include "pipeline/Model.h"

namespace tomcat {
    namespace model {

        /**
         * Class responsible to save a model's parameters to files in a specific
         * folder.
         */
        class ModelSaver {
          public:
            //------------------------------------------------------------------
            // Constructors & Destructor
            //------------------------------------------------------------------

            /**
             * Creates an instance of the saver.
             *
             * @param model: Model
             * @param output_folder_path: folder where the model's parameters
             * must be saved in
             */
            ModelSaver(const ModelPtr & model,
                     const std::string& output_folder_path);

            ~ModelSaver() = default;

            //------------------------------------------------------------------
            // Copy & Move constructors/assignments
            //------------------------------------------------------------------

            ModelSaver(const ModelSaver& saver);

            ModelSaver& operator=(const ModelSaver& saver);

            ModelSaver(ModelSaver&&) = default;

            ModelSaver& operator=(ModelSaver&&) = default;

            //------------------------------------------------------------------
            // Virtual functions
            //------------------------------------------------------------------

            /**
             * Prepares the trainer to a series of calls to the function fit by
             * performing necessary cleanups.
             */
            virtual void prepare();

            /**
             * Saves a model's parameters into files in a specific folder.
             */
            virtual void save();

          protected:
            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            /**
             * Copies data members from another ModelLoader.
             */
            void copy(const ModelSaver& saver);

            //------------------------------------------------------------------
            // Data members
            //------------------------------------------------------------------
            ModelPtr model;

            // Folder where the model's parameters' files will be saved.
            std::string output_folder_path;

            // Cross validation step. This is incremented at each call of the
            // function save. It can be used to identify a folder with
            // parameters for a specific cross validation step.
            int cv_step = 0;
        };

    } // namespace model
} // namespace tomcat
