#pragma once

#include <memory>
#include <vector>

namespace tomcat {
    namespace model {
        class Node;
        class RandomVariableNode;

        typedef std::shared_ptr<RandomVariableNode> NodePtr;
        typedef std::shared_ptr<RandomVariableNode> RVNodePtr;
        typedef std::vector<NodePtr> NodePtrVec;
        typedef std::vector<RVNodePtr> NodePtrVec;
    }
} // namespace tomcat
