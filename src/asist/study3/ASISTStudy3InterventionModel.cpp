#include "ASISTStudy3InterventionModel.h"

#include <boost/filesystem.hpp>
#include <fmt/format.h>

#include "asist/study3/ASISTStudy3MessageConverter.h"
#include "pgm/cpd/PoissonCPD.h"
#include "utils/Definitions.h"
#include "utils/JSONChecker.h"

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors
        //----------------------------------------------------------------------
        ASISTStudy3InterventionModel::ASISTStudy3InterventionModel(
            const nlohmann::json& json_settings) {
            this->parse_settings(json_settings);
            this->create_components();
        }

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        void
        ASISTStudy3InterventionModel::save_to(const string& output_dir) const {
            boost::filesystem::create_directories(output_dir);
        }

        void ASISTStudy3InterventionModel::load_from(const string& input_dir,
                                                     bool freeze_model) {}

        unique_ptr<Model> ASISTStudy3InterventionModel::clone() const {
            ASISTStudy3InterventionModel new_model;

            new_model.encouragement_node = unique_ptr<RandomVariableNode>(
                dynamic_cast<RandomVariableNode*>(
                    this->encouragement_node->clone().get()));

            return make_unique<ASISTStudy3InterventionModel>(move(new_model));
        }

        void ASISTStudy3InterventionModel::parse_settings(
            const nlohmann::json& json_settings) {

            this->mean_encouragement = json_settings["mean_encouragement"];
        }

        void ASISTStudy3InterventionModel::create_components() {
            this->create_motivation_component();
        }

        void ASISTStudy3InterventionModel::create_motivation_component() {
            NodeMetadata metadata =
                NodeMetadata::create_single_time_link_metadata(
                    ASISTStudy3MessageConverter::Labels::ENCOURAGEMENT,
                    false,
                    true,
                    0,
                    1,
                    1);
            this->encouragement_node = make_shared<RandomVariableNode>(
                make_shared<NodeMetadata>(move(metadata)), 0);
            PoissonCPD cpd(
                {}, Eigen::VectorXd::Constant(1, this->mean_encouragement));
            this->encouragement_node->set_cpd(make_shared<PoissonCPD>(cpd));
        }

        //----------------------------------------------------------------------
        // Getters & Setters
        //----------------------------------------------------------------------

        const shared_ptr<RandomVariableNode>&
        ASISTStudy3InterventionModel::get_encouragement_node() const {
            return encouragement_node;
        }
    } // namespace model
} // namespace tomcat
