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
        class DynamicBayesNet;
        class TimerNode;

        typedef std::shared_ptr<NodeMetadata> MetadataPtr;
        typedef std::shared_ptr<Node> NodePtr;
        typedef std::shared_ptr<RandomVariableNode> RVNodePtr;
        typedef std::shared_ptr<TimerNode> TimerNodePtr;
        typedef std::vector<NodePtr> NodePtrVec;
        typedef std::vector<RVNodePtr> RVNodePtrVec;
        typedef std::vector<TimerNode> TimerNodePtrVec;
        typedef std::shared_ptr<CPD> CPDPtr;
        typedef std::shared_ptr<Distribution> DistributionPtr;
        typedef std::vector<DistributionPtr> DistributionPtrVec;
        typedef std::shared_ptr<DynamicBayesNet> DBNPtr;

        // Exact Inference
        class MessageNode;
        class FactorNode;
        class VariableNode;
        class SegmentExpansionFactorNode;
        class SegmentTransitionFactorNode;
        class SegmentMarginalizationFactorNode;

        typedef std::shared_ptr<MessageNode> MsgNodePtr;
        typedef std::vector<MsgNodePtr> MsgNodePtrVec;
        typedef std::shared_ptr<FactorNode> FactorNodePtr;
        typedef std::vector<FactorNodePtr> FactorNodePtrVec;
        typedef std::shared_ptr<VariableNode> VarNodePtr;
        typedef std::vector<VarNodePtr> VarNodePtrVec;
        typedef std::shared_ptr<SegmentExpansionFactorNode> SegExpFactorNodePtr;
        typedef std::vector<SegExpFactorNodePtr> SegExpFactorNodePtrVec;
        typedef std::shared_ptr<SegmentTransitionFactorNode>
            SegTransFactorNodePtr;
        typedef std::vector<SegTransFactorNodePtr> SegTransFactorNodePtrVec;
        typedef std::shared_ptr<SegmentMarginalizationFactorNode>
            SegMarFactorNodePtr;
        typedef std::vector<SegMarFactorNodePtr> SegMarFactorNodePtrVec;

        // General
        class Estimator;
        class MessageConverter;
        class Agent;

        typedef std::shared_ptr<Estimator> EstimatorPtr;
        typedef std::vector<EstimatorPtr> EstimatorPtrVec;
        typedef std::shared_ptr<MessageConverter> MsgConverterPtr;
        typedef std::vector<MsgConverterPtr> MsgConverterPtrVec;
        typedef std::shared_ptr<Agent> AgentPtr;
        typedef std::vector<AgentPtr> AgentPtrVec;
    } // namespace model
} // namespace tomcat
