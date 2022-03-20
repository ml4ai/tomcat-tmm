#pragma once

#include <memory>
#include <string>

#include "DBNTrainer.h"

#include "utils/Definitions.h"
#include "pgm/DynamicBayesNet.h"
#include "pipeline/training/ModelLoader.h"

namespace tomcat {
    namespace model {

        /**
         * Class responsible for loading a model from a pre-trained set of
         * parameters.
         */
        class DBNLoader : public DBNTrainer {
          public:
            //------------------------------------------------------------------
            // Constructors & Destructor
            //------------------------------------------------------------------

            /**
             * Creates an instance of the loader.
             *
             * @param model: DBN
             * @param input_folder_path: folder where the model's parameters
             * are stored
             * @param freeze_loaded_nodes: whether loaded nodes should be frozen
             */
            DBNLoader(const std::shared_ptr<DynamicBayesNet>& model,
                      const std::string& input_folder_path,
                      bool freeze_loaded_nodes = true);

            ~DBNLoader();

            //------------------------------------------------------------------
            // Copy & Move constructors/assignments
            //------------------------------------------------------------------
            DBNLoader(const DBNLoader& loader);

            DBNLoader& operator=(const DBNLoader& loader);

            DBNLoader(DBNLoader&&) = default;

            DBNLoader& operator=(DBNLoader&&) = default;

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
             * Loads parameter samples generated during the training process.
             */
            void load_partials();

            /**
             * Copies data members from another ModelLoader.
             */
            void copy(const DBNLoader& loader);

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
            bool freeze_loaded_nodes;

        };

    } // namespace model
} // namespace tomcat
