#include "ModelSaver.h"

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
        ModelSaver::ModelSaver(const ModelPtr& model,
                               const string& output_folder_path)
            : model(model), output_folder_path(output_folder_path) {}

        //----------------------------------------------------------------------
        // Copy & Move constructors/assignments
        //----------------------------------------------------------------------
        ModelSaver::ModelSaver(const ModelSaver& saver) {
            this->copy(saver);
        }

        ModelSaver& ModelSaver::operator=(const ModelSaver& saver) {
            this->copy(saver);
            return *this;
        }

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        void ModelSaver::copy(const ModelSaver& saver) {
            this->model = saver.model;
            this->output_folder_path = saver.output_folder_path;
            this->cv_step = saver.cv_step;
        }

        void ModelSaver::prepare() { this->cv_step = 0; }

        void ModelSaver::save() {
            // If the name of the folder has a placeholder for the cv step,
            // replace it with the current number.
            const string final_folder_path =
                fmt::format(this->output_folder_path, this->cv_step + 1);
            this->model->save_to(final_folder_path);

            this->cv_step++;
        }

    } // namespace model
} // namespace tomcat
