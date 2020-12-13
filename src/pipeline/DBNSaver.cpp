#include "DBNSaver.h"

#include <fmt/format.h>
#include <boost/filesystem.hpp>

#include "utils/FileHandler.h"

namespace tomcat {
    namespace model {

        using namespace std;
        namespace fs = boost::filesystem;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        DBNSaver::DBNSaver(shared_ptr<DynamicBayesNet> model,
                           shared_ptr<DBNTrainer> trainer,
                           string output_folder_path,
                           bool include_partials)
            : model(model), trainer(trainer),
              output_folder_path(output_folder_path),
              include_partials(include_partials) {}

        DBNSaver::~DBNSaver() {}

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        void DBNSaver::prepare() { this->cv_step = 0; }

        void DBNSaver::save() {
            // If the name of the folder has a placeholder for the cv step,
            // replace it with the current number.
            const string final_folder_path =
                fmt::format(this->output_folder_path, this->cv_step + 1);
            this->model->save_to(final_folder_path);

            if (this->include_partials) {
                this->save_partials(final_folder_path);
            }

            this->cv_step++;
        }

        void DBNSaver::save_partials(const string& model_dir)
        const {
            const string partials_dir =
                fmt::format("{}/partials", model_dir);
            fs::create_directories(partials_dir);

            unordered_map<string, Tensor3> partials =
                this->trainer->get_partials(this->cv_step);

            for (const auto& [param_label, param_samples] : partials) {
                const string filepath =
                    fmt::format("{}/{}", partials_dir, param_label);
                save_tensor_to_file(filepath, param_samples);
            }
        }

    } // namespace model
} // namespace tomcat
