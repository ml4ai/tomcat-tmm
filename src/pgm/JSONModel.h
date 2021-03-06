#pragma once

#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <memory>
#include <string>

#include <nlohmann/json.hpp>
#include "utils/Definitions.h"

/**
 * This file creates a DBN from a the model's specifications in a JSON file.
 */

namespace tomcat {
    namespace model {

        class DynamicBayesNet;

        typedef std::unordered_map<std::string, std::vector<MetadataPtr>>
            MetadataMap;
        typedef std::unordered_map<std::string, RVNodePtrVec> RVMap;
        typedef std::unordered_set<std::string> NodeSet;
        typedef std::unordered_map<std::string, std::pair<int, int>>
            ParamMapConfig;

        DynamicBayesNet create_model_from_json(const std::string& filepath);

    } // namespace model
} // namespace tomcat