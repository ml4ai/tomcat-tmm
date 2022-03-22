#include "Model.h"

#include "asist/study3/ASISTStudy3InterventionModel.h"

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Static functions
        //----------------------------------------------------------------------
        ModelPtr Model::factory(const std::string& model_name,
                                const nlohmann::json& json_settings) {
            if (model_name == ASISTStudy3InterventionModel::NAME) {
                return make_shared<ASISTStudy3InterventionModel>(json_settings);
            }

            return nullptr;
        }

    } // namespace model
} // namespace tomcat
