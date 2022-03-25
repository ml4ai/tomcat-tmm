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
        DBNLoader::DBNLoader(const shared_ptr<DynamicBayesNet>& model,
                             const string& input_folder_path,
                             bool freeze_loaded_nodes)
            : DBNTrainer(model), input_folder_path(input_folder_path),
              freeze_loaded_nodes(freeze_loaded_nodes) {}

        DBNLoader::~DBNLoader() {}

        //----------------------------------------------------------------------
        // Copy & Move constructors/assignments
        //----------------------------------------------------------------------
        DBNLoader::DBNLoader(const DBNLoader& loader)
            : DBNTrainer(dynamic_pointer_cast<DynamicBayesNet>(loader.model)),
              freeze_loaded_nodes(loader.freeze_loaded_nodes) {
            this->copy(loader);
        }

        DBNLoader& DBNLoader::operator=(const DBNLoader& loader) {
            this->copy(loader);
            return *this;
        }

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        void DBNLoader::copy(const DBNLoader& loader) {
            this->input_folder_path = loader.input_folder_path;
            this->split_idx = loader.split_idx;
            this->freeze_loaded_nodes = loader.freeze_loaded_nodes;
            this->param_label_to_samples = loader.param_label_to_samples;
        }

        void DBNLoader::prepare() {
            DBNTrainer::prepare();
            this->split_idx = 0;
        }

        void DBNLoader::fit(const EvidenceSet& training_data) {
            // If the name of the folder has a placeholder for the cv step,
            // replace it with the current number.

            string final_folder_path =
                fmt::format(this->input_folder_path, this->split_idx + 1);
            if (fs::exists(final_folder_path)) {
                this->model->load_from(final_folder_path,
                                       this->freeze_loaded_nodes);
                this->load_partials();
                this->split_idx++;
            }
            else {
                stringstream ss;
                ss << "The directory " << final_folder_path
                   << " does not exist.";
                throw TomcatModelException(ss.str());
            }
        }

        void DBNLoader::load_partials() {
            this->param_label_to_samples.clear();
            const string partials_dir =
                fmt::format("{}/{}", this->input_folder_path, "partials");
            if (fs::exists(partials_dir)) {
                this->param_label_to_samples.push_back(
                    unordered_map<string, Tensor3>());
                for (const auto& file : fs::directory_iterator(partials_dir)) {
                    string filename = file.path().filename().string();
                    if (fs::is_regular_file(file)) {
                        const string param_label = remove_extension(filename);
                        if (dynamic_pointer_cast<DynamicBayesNet>(this->model)
                                ->has_parameter_node_with_label(param_label)) {
                            const string filepath =
                                get_filepath(partials_dir, param_label);
                            Tensor3 param_samples =
                                read_tensor_from_file(filepath);
                            this->param_label_to_samples[this->split_idx]
                                                        [param_label] =
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

    } // namespace model
} // namespace tomcat
