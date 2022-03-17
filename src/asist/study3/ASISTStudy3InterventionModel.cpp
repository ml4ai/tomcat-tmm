#include "ASISTStudy3InterventionModel.h"

#include <boost/filesystem.hpp>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <fmt/format.h>

#include "pgm/cpd/HistogramCPD.h"
#include "utils/Definitions.h"

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        void
        ASISTStudy3InterventionModel::save_to(const string& output_dir) const {
            boost::filesystem::create_directories(output_dir);
        }

        void ASISTStudy3InterventionModel::load_from(const string& input_dir,
                                                     bool freeze_model) {

            string encouragement_samples_path =
                fmt::format("{}/EncouragementSamples", input_dir);
            if (boost::filesystem::exists(
                    boost::filesystem::path(encouragement_samples_path))) {

                Eigen::MatrixXd samples =
                    read_matrix_from_file(encouragement_samples_path);
                HistogramCPD cpd({}, {samples.row(0)});
                this->encouragement_node->set_cpd(
                    make_shared<HistogramCPD>(cpd));
            }
            else {
                throw TomcatModelException(
                    "Encouragement samples file not found.");
            }
        }

        unique_ptr<Model> ASISTStudy3InterventionModel::clone() const {
            ASISTStudy3InterventionModel new_model;

            new_model.encouragement_node = unique_ptr<RandomVariableNode>(
                dynamic_cast<RandomVariableNode*>(
                    this->encouragement_node->clone().get()));

            return make_unique<ASISTStudy3InterventionModel>(move(new_model));
        }

        void ASISTStudy3InterventionModel::create_components() {
            this->create_motivation_component();
        }

        void ASISTStudy3InterventionModel::create_motivation_component() {
            NodeMetadata metadata =
                NodeMetadata::create_single_time_link_metadata(
                    "Encouragement", false, true, 0, 1, 1);
            this->encouragement_node = make_unique<RandomVariableNode>(
                make_shared<NodeMetadata>(move(metadata)), 0);
        }

    } // namespace model
} // namespace tomcat
