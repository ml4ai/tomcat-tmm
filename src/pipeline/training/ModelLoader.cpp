#include "ModelLoader.h"

#include <boost/filesystem.hpp>
#include <fmt/format.h>

#include "utils/FileHandler.h"
#include "utils/Tensor3.h"

namespace tomcat {
    namespace model {

        using namespace std;
        namespace fs = boost::filesystem;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        ModelLoader::ModelLoader(const ModelPtr& model,
                                 const string& input_folder_path,
                                 bool freeze_model)
            : ModelTrainer(model), input_folder_path(input_folder_path),
              freeze_model(freeze_model) {}

        //----------------------------------------------------------------------
        // Copy & Move constructors/assignments
        //----------------------------------------------------------------------
        ModelLoader::ModelLoader(const ModelLoader& loader)
            : ModelTrainer(model) {
            this->copy(loader);
        }

        ModelLoader& ModelLoader::operator=(const ModelLoader& loader) {
            this->copy(loader);
            return *this;
        }

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        void ModelLoader::copy(const ModelLoader& loader) {
            this->model = loader.model;
            this->input_folder_path = loader.input_folder_path;
            this->split_idx = loader.split_idx;
            this->freeze_model = loader.freeze_model;
        }

        void ModelLoader::prepare() {
            this->split_idx = 0;
        }

        void ModelLoader::fit(const EvidenceSet& training_data) {
            // If the name of the folder has a placeholder for the cv step,
            // replace it with the current number.

            string final_folder_path =
                fmt::format(this->input_folder_path, this->split_idx + 1);
            if (fs::exists(final_folder_path)) {
                this->model->load_from(final_folder_path,
                                       this->freeze_model);
                this->split_idx++;
            }
            else {
                stringstream ss;
                ss << "The directory " << final_folder_path
                   << " does not exist.";
                throw TomcatModelException(ss.str());
            }
        }

        void ModelLoader::get_info(nlohmann::json& json) const {
            json["type"] = "pre_trained";
            json["input_folder_path"] = this->input_folder_path;
        }

    } // namespace model
} // namespace tomcat
