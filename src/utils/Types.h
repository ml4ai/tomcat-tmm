#pragma once

#include <memory>
#include <vector>

namespace tomcat {
    namespace model {
        class Node;
        class RandomVariableNode;
        class NodeMetadata;
        class CPD;
        class Distribution;

        typedef std::shared_ptr<NodeMetadata> MetadataPtr;
        typedef std::shared_ptr<Node> NodePtr;
        typedef std::shared_ptr<RandomVariableNode> RVNodePtr;
        typedef std::vector<NodePtr> NodePtrVec;
        typedef std::vector<RVNodePtr> RVNodePtrVec;
        typedef std::shared_ptr<CPD> CPDPtr;
        typedef std::shared_ptr<Distribution> DistributionPtr;
        typedef std::vector<DistributionPtr> DistributionPtrVec;

        // Exact Inference
        class MessageNode;
        class FactorNode;
        class VariableNode;

        typedef std::shared_ptr<MessageNode> MsgNodePtr;
        typedef std::vector<MsgNodePtr> MsgNodePtrVec;
        typedef std::shared_ptr<FactorNode> FactorNodePtr;
        typedef std::vector<FactorNodePtr> FactorNodePtrVec;
        typedef std::shared_ptr<VariableNode> VarNodePtr;
        typedef std::vector<VarNodePtr> VarNodePtrVec;
    } // namespace model
} // namespace tomcat
