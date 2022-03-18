#include "JSONChecker.h"

#include "fmt/format.h"

namespace tomcat {
    namespace model {

        using namespace std;

        void check_field(const nlohmann::json& json, const string& field) {
            if (!EXISTS(field, json)) {
                throw TomcatModelException(fmt::format("Field {} not found."));
            }
        }

    } // namespace model
} // namespace tomcat
