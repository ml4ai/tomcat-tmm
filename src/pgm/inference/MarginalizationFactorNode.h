#pragma once

#include <utility>

#include "pgm/inference/FactorNode.h"

namespace tomcat {
    namespace model {

        /**
         * Represents a factor node that marginalizes joint probabilities to
         * compute the probability of one of the nodes involved in the joint
         * probability.
         */
        class MarginalizationFactorNode : public FactorNode {
          public:
            //------------------------------------------------------------------
            // Constructors & Destructor
            //------------------------------------------------------------------

            /**
             * Creates an instance of the marginalization factor node.
             *
             * @param label: label of the node that gives raise to segments
             * @param time_step: time step of this template node
             * @param joint_ordering_map: order and cardinality of the nodes 
             * in the joint probability
             * @param joint_node_label: label of the joint node this factor
             * marginalizes or produces
             */
            MarginalizationFactorNode(
                const std::string& label,
                int time_step,
                const CPD::TableOrderingMap& joint_ordering_map,
                const std::string& joint_node_label);

            ~MarginalizationFactorNode();

            //------------------------------------------------------------------
            // Copy & Move constructors/assignments
            //------------------------------------------------------------------

            MarginalizationFactorNode(
                const MarginalizationFactorNode& factor_node);

            MarginalizationFactorNode&
            operator=(const MarginalizationFactorNode& factor_node);

            MarginalizationFactorNode(
                MarginalizationFactorNode&&) = default;

            MarginalizationFactorNode&
            operator=(MarginalizationFactorNode&&) = default;

            //------------------------------------------------------------------
            // Static functions
            //------------------------------------------------------------------

            /**
             * Adds an identifier to the label to indicate this is a
             * marginalization node.
             *
             * @param original_label: original label
             *
             * @return Stamped label
             */
            static std::string compose_label(const std::string& original_label);

            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            /**
             * Marginalizes out other nodes in the joint distribution to
             * compute the probability of a target node.
             *
             * @param template_target_node: template instance of the node where
             * the message should go to
             * @param template_time_step: time step of this node where to get
             * the incoming messages from. If the template node belongs to the
             * repeatable structure, this information is needed to know which
             * time step to address to retrieve the incoming messages.
             * @param target_time_step: real time step of the target node
             * @param direction: direction of the message passing
             *
             * @return Message
             */
            Tensor3 get_outward_message_to(
                const std::shared_ptr<MessageNode>& template_target_node,
                int template_time_step,
                int target_time_step,
                Direction direction) const override;

            bool is_segment() const override;

          private:
            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            /**
             * Copies data members from another factor node.
             *
             * @param node: other factor node
             */
            void copy_node(const MarginalizationFactorNode& node);

            //------------------------------------------------------------------
            // Data members
            //------------------------------------------------------------------

            int joint_cardinality = 1;

            std::string joint_node_label;

        };

    } // namespace model
} // namespace tomcat
