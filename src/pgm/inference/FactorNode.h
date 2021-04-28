#pragma once

#include <vector>

#include "MessageNode.h"

#include "distribution/Distribution.h"
#include "pgm/cpd/CPD.h"
#include "utils/Definitions.h"

namespace tomcat {
    namespace model {

        /**
         * This class represents a factor node in a factor graph.
         */
        class FactorNode : public MessageNode {
          public:
            //------------------------------------------------------------------
            // Structs
            //------------------------------------------------------------------
            struct PotentialFunction {
                // The joint CPD table is implemented as a single matrix where
                // the rows are combinations of parents' assignments for a node
                // and the columns are the node's assignments. We need to now
                // the index of each parent in this table to be able to index
                // them correctly when performing reductions or transformations
                // in the matrix. This information is stored in the ordering
                // map.
                CPD::TableOrderingMap ordering_map;

                // CPD table
                Eigen::MatrixXd probability_table;

                // In some scenarios, we won't be able to enumerate all the
                // probabilities in a table. The list of distributions of
                // a CPD will be needed instead.
                DistributionPtrVec distributions;

                // Node's label in P(Node | ...)
                std::string main_node_label;

                // The node's label can be the same as one of its parents. For
                // instance, if the CPD table defines a transition matrix. The
                // labels are the same but the time step. The matrix will need
                // to be rotated to deal with backward messages, so this
                // attribute keeps track of duplicate keys so the matrix can be
                // correctly indexed even when rotated. In this context, a
                // rotation means replacing the node position in the matrix  by
                // one of it's parents. Suppose a CPD defines the probability
                // P(S|S,Q), the rotation P(Q|S,S) will cause a problem if
                // duplicate keys are not correctly handled.
                std::string duplicate_key = "";

                PotentialFunction() {}

                PotentialFunction(const CPD::TableOrderingMap& ordering_map,
                                  const Eigen::MatrixXd& probability_table,
                                  const std::string main_node_label)
                    : ordering_map(ordering_map),
                      probability_table(probability_table),
                      main_node_label(main_node_label) {}

                PotentialFunction(
                    const CPD::TableOrderingMap& ordering_map,
                    const DistributionPtrVec & distributions,
                    const std::string main_node_label)
                    : ordering_map(ordering_map), distributions(distributions),
                      main_node_label(main_node_label) {}

                static std::string
                get_alternative_key_label(const std::string& label) {
                    return label + "*";
                }
            };

            //------------------------------------------------------------------
            // Constructors & Destructor
            //------------------------------------------------------------------

            // Default constructor
            FactorNode();

            /**
             * Creates an instance of a factor node.
             *
             * @param label: node's label. The factor's label will be a
             * modified version of the label informed to indicate that it's a
             * factor.
             * @param time_step: factor's time step
             * @param probability_table: matrix representing the potential
             * function
             * @param ordering_map: potential function matrix's ordering map
             */
            FactorNode(const std::string& label,
                       int time_step,
                       const Eigen::MatrixXd& probability_table,
                       const CPD::TableOrderingMap& ordering_map,
                       const std::string& cpd_node_label);

            /**
             * Creates an instance of a factor node.
             *
             * @param label: node's label. The factor's label will be a
             * modified version of the label informed to indicate that it's a
             * factor.
             * @param distributions: list of distributions of a CPD that
             * defines the factor node's potential function
             * @param time_step: factor's time step
             * @param ordering_map: potential function matrix's ordering map
             */
            FactorNode(const std::string& label,
                       int time_step,
                       const DistributionPtrVec& distributions,
                       const CPD::TableOrderingMap& ordering_map,
                       const std::string& cpd_node_label);

            ~FactorNode();

            //------------------------------------------------------------------
            // Copy & Move constructors/assignments
            //------------------------------------------------------------------
            FactorNode(const FactorNode& node);

            FactorNode& operator=(const FactorNode& node);

            FactorNode(FactorNode&&) = default;

            FactorNode& operator=(FactorNode&&) = default;

            //------------------------------------------------------------------
            // Static functions
            //------------------------------------------------------------------

            /**
             * Adds a prefix to a label to indicate that it's a factor node's
             * label.
             *
             * @param original_label: label used in the composition.
             *
             * @return Factor node label
             */
            static std::string compose_label(const std::string& original_label);

            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            /**
             * Computes the product of all the incoming messages (except the one
             * coming from the target node) and the potential function. Next,
             * marginalizes out the incoming nodes and returns the resultant
             * message.
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

            bool is_factor() const override;

            bool is_segment() const override;

            /**
             * Creates a potential function (and associated rotations) by
             * having a binary distribution table, where the column 0 contains
             * p(x != value) and column 1 contains p(x == value).
             *
             * @param value: value to preserve the probabilities
             */
            void create_aggregate_potential(int value);

            /**
             * Sets the aggregate potential for a given value as the working
             * one.
             *
             * @param value: key value of a previously created aggregate
             * potential
             */
            void use_aggregate_potential(int value);

            /**
             * Uses the node's original potential. The one without aggregation.
             */
            void use_original_potential();

          protected:
            //------------------------------------------------------------------
            // Structs
            //------------------------------------------------------------------
            struct Potential {
                PotentialFunction potential;

                // The potential function here is represented as a matrix that
                // corresponds to the conditional probability of a child node
                // given its parents. When calculating messages, depending on
                // the direction the message flows, the potential function
                // matrix needs to be rotated so that the child become one of
                // the parents. That's what this adjusted table is for. It will
                // store all possible rotations of the potential function matrix
                // according to the the node that should assume the child's
                // position.
                std::unordered_map<std::string, PotentialFunction>
                    node_label_to_rotated_potential;
            };

            struct AggregatePotential {
                // The following data structures store potential functions
                // aggregated into binary distributions. The key to the map is
                // the assignment preserved (value 1 in the modified potential
                // function). This will be used by a message passing estimator
                // to make predictions about a certain assignment.

                std::unordered_map<int, PotentialFunction> potential;

                std::unordered_map<
                    int,
                    std::unordered_map<std::string, PotentialFunction>>
                    node_label_to_rotated_potential;
            };

            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            /**
             * Returns all possible rotations of a potential function.
             *
             * @param original_potential: potential function to rotate
             */
            std::unordered_map<std::string, FactorNode::PotentialFunction>
            create_potential_function_rotations(
                const PotentialFunction& original_potential);

            /**
             * Copies data members from another factor node.
             *
             * @param node: other factor node
             */
            void copy_node(const FactorNode& node);

            /**
             * Returns the messages that arrive into the factor node (ignoring a
             * given target node) in the order defined by its parents nodes in
             * the potential function ordering map. This is crucial to correclty
             * multiply incoming messages with the potential function matrix in
             * a correct way.
             *
             * @param ignore_label: label of the node that must be ignored. If
             * there's any, messages from this node in the target time step must
             * be discarded as this node is the final destination of the
             * messages we aim to compute when calling this function.
             * @param template_time_step: time step of this node where to get
             * the incoming messages from. If the template node belongs to the
             * repeatable structure, this information is needed to know which
             * time step to address to retrieve the incoming messages.
             * @param target_time_step: real time step of the target node
             * @param potential_function: potential function
             *
             * @return Incoming messages
             */
            std::vector<Tensor3> get_incoming_messages_in_order(
                const std::string& ignore_label,
                int template_time_step,
                int target_time_step,
                const PotentialFunction& potential_function) const;

            /**
             * Rotates a table to move one of the indexing nodes to the
             * columns of the table.
             *
             * @param table: original table
             * @param main_node_cardinality: cardinality of the node to be
             * put in evidence (placed in the columns of the rotated table)
             * @param right_cumulative_cardinality: number of all possible
             * assignment combinations of the indexing nodes to the right of
             * the main node in the original table indexing scheme
             *
             * @return Rotated table
             */
            Eigen::MatrixXd
            rotate_table(const Eigen::MatrixXd& table,
                         int main_node_cardinality,
                         int right_cumulative_cardinality) const;

            /**
             * Returns a tensor formed by the cartesian product of each row
             * in a collection of tensors.
             *
             * @param tensors: tensors
             *
             * @return cartesian product of rows from a collection of tensors
             * . The final tensor will be formed by matrices with the same
             * number of rows and columns given by the cartesian product of
             * the rows in the collection of tensors.
             */
            Tensor3
            get_cartesian_tensor(const std::vector<Tensor3>& tensors) const;

            //------------------------------------------------------------------
            // Data members
            //------------------------------------------------------------------
            Potential original_potential;

            AggregatePotential aggregate_potential;

            // Potentials used in a given moment. Either original or aggregated.
            Potential working_potential;
        };

    } // namespace model
} // namespace tomcat
