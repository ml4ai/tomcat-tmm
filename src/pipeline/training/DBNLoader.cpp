#include "DBNLoader.h"

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
        DBNLoader::DBNLoader(shared_ptr<DynamicBayesNet> model,
                             string input_folder_path,
                             bool freeze_loaded_nodes)
            : model(model), input_folder_path(input_folder_path),
              freeze_loaded_nodes(freeze_loaded_nodes) {}

        DBNLoader::~DBNLoader() {}

        //----------------------------------------------------------------------
        // Copy & Move constructors/assignments
        //----------------------------------------------------------------------
        DBNLoader::DBNLoader(const DBNLoader& loader) {
            this->copy_loader(loader);
        }

        DBNLoader& DBNLoader::operator=(const DBNLoader& loader) {
            this->copy_loader(loader);
            return *this;
        }

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        void DBNLoader::copy_loader(const DBNLoader& loader) {
            this->model = loader.model;
            this->input_folder_path = loader.input_folder_path;
            this->cv_step = loader.cv_step;
            this->freeze_loaded_nodes = loader.freeze_loaded_nodes;
            this->param_label_to_samples = loader.param_label_to_samples;
        }

        void DBNLoader::prepare() { this->cv_step = 0; }

        void DBNLoader::fit(const EvidenceSet& training_data) {
            // If the name of the folder has a placeholder for the cv step,
            // replace it with the current number.
            string final_folder_path =
                fmt::format(this->input_folder_path, ++this->cv_step);
            this->model->load_from(final_folder_path, this->freeze_loaded_nodes);
            this->load_partials();
        }

        void DBNLoader::load_partials() {
            this->param_label_to_samples.clear();
            const string partials_dir =
                fmt::format("{}/{}", this->input_folder_path, "partials");
            if (fs::exists(partials_dir)) {
                for (const auto& file : fs::directory_iterator(partials_dir)) {
                    string filename = file.path().filename().string();
                    if (fs::is_regular_file(file)) {
                        const string param_label = remove_extension(filename);
                        if (this->model->has_parameter_node_with_label(
                                param_label)) {
                            const string filepath = get_filepath(
                                partials_dir, param_label + ".txt");
                            Tensor3 param_samples =
                                read_tensor_from_file(filepath);
                            this->param_label_to_samples[param_label] =
                                param_samples;
                        }
                    }
                }
            }
        }

        void DBNLoader::get_info(nlohmann::json& json) const {
            json["type"] = "pre_trained";
            json["input_folder_path"] = this->input_folder_path;
        }

        shared_ptr<DynamicBayesNet> DBNLoader::get_model() const {
            return this->model;
        }

    } // namespace model
} // namespace tomcat
