#include "DBNSaver.h"

#include <boost/filesystem.hpp>
#include <fmt/format.h>

#include "utils/FileHandler.h"

namespace tomcat {
    namespace model {

        using namespace std;
        namespace fs = boost::filesystem;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        DBNSaver::DBNSaver(const shared_ptr<DynamicBayesNet>& model,
                           const string& output_folder_path,
                           const shared_ptr<DBNTrainer>& trainer,
                           bool include_partials)
            : ModelSaver(model, output_folder_path), trainer(trainer),
              include_partials(include_partials) {}

        //----------------------------------------------------------------------
        // Copy & Move constructors/assignments
        //----------------------------------------------------------------------
        DBNSaver::DBNSaver(const DBNSaver& saver)
            : ModelSaver(saver.model, saver.output_folder_path) {
            ModelSaver::copy(saver);
            this->trainer = saver.trainer;
            this->include_partials = saver.include_partials;
        }

        DBNSaver& DBNSaver::operator=(const DBNSaver& saver) {
            ModelSaver::copy(saver);
            this->trainer = saver.trainer;
            this->include_partials = saver.include_partials;
            return *this;
        }

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------

        void DBNSaver::save() {
            if (this->include_partials) {
                const string final_folder_path =
                    fmt::format(this->output_folder_path, this->cv_step + 1);
                this->save_partials(final_folder_path);
            }

            ModelSaver::save();
        }

        void DBNSaver::save_partials(const string& model_dir) const {
            const string partials_dir = fmt::format("{}/partials", model_dir);
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
