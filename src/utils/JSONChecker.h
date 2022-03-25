#pragma once

#include <nlohmann/json.hpp>
#include "Definitions.h"

namespace tomcat {
    namespace model {

        void check_field(const nlohmann::json& json, const std::string& field);

    } // namespace model
} // namespace tomcat
