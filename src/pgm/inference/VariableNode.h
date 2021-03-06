#pragma once

#include <unordered_set>

#include "MessageNode.h"

#include "utils/Definitions.h"

namespace tomcat {
    namespace model {

        /**
         * This class represents a variable node in a factor graph.
         */
        class VariableNode : public MessageNode {
          public:
            //------------------------------------------------------------------
            // Constructors & Destructor
            //------------------------------------------------------------------

            /**
             * Creates an instance of a variable node.
             *
             * @param label: node's label
             * @param time_step: node's time step
             * @param cardinality: node's cardinality
             */
            VariableNode(const std::string& label,
                         int time_step,
                         int cardinality);

            /**
             * Creates an instance of a segment variable node.
             *
             * @param label: node's label
             * @param time_step: node's time step
             */
            VariableNode(const std::string& label, int time_step);

            ~VariableNode();

            //------------------------------------------------------------------
            // Copy & Move constructors/assignments
            //------------------------------------------------------------------
            VariableNode(const VariableNode& node);

            VariableNode& operator=(const VariableNode& node);

            VariableNode(VariableNode&&) = default;

            VariableNode& operator=(VariableNode&&) = default;

            //------------------------------------------------------------------
            // Static functions
            //------------------------------------------------------------------

            /**
             * Adds a prefix to a label to indicate that it's a segment of a
             * time controlled node.
             *
             * @param original_label: label of the time controlled node.
             *
             * @return Segment label for a time controlled node.
             */
            static std::string
            compose_segment_label(const std::string& timed_node_label);

            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            /**
             * Computes the element-wise product of all the incoming messages
             * excluding the one that comes from the target node.
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
             * Computes the marginal distribution for a given node in a certain
             * point in time. The marginal is given by the multiplication of all
             * the incoming messages to the node.
             *
             * @param time_step: time step for inference
             * @param normalized: whether the multiplication of incoming
             * messages must be normalized or not.
             *
             * @return Marginal distribution
             */
            Eigen::MatrixXd get_marginal_at(int time_step,
                                            bool normalized) const;

            /**
             * Stores data for the node in a given time step. If there's data in
             * a given time step, messages from this node in that time step are
             * deterministic.
             *
             * @param time_step: time step where the data must be set
             * @param data: vector with node's values from several data sets
             *
             */
            void set_data_at(int time_step,
                             const Eigen::MatrixXd& data);

            /**
             * If there's data assigned to a node in a given time step, remove
             * it from there.
             *
             * @param time_step: time step to remove the data from
             *
             */
            void erase_data_at(int time_step);

            /**
             * Adds an income node to the set of nodes to be ignored when
             * producing a backward message to a certain target.
             *
             * @param incoming_node_label: label of the incoming node to be
             * ignored
             * @param target_node_label: name of the target node to which the
             * message is being produced
             */
            void add_backward_blocking(const std::string& incoming_node_label,
                                       const std::string& target_node_label);

            /**
             * Aggregates the probabilities of the values different than the
             * provided assignment value.
             *
             * @param node_assignment_value: value to be preserved.
             */
            void aggregate(int node_assignment_value);

            /**
             * Incoming messages won't be aggregated.
             */
            void do_not_aggregate();

            //------------------------------------------------------------------
            // Getters & Setters
            //------------------------------------------------------------------
            int get_cardinality() const;

          private:
            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            /**
             * Copies data members from another variable node.
             *
             * @param node: other variable node
             */
            void copy_node(const VariableNode& node);

            //------------------------------------------------------------------
            // Data members
            //------------------------------------------------------------------
            int cardinality;

            // Stores observable data for the node per time slice. The rows of
            // the data matrix contains a one-hot vector encode representing the
            // value observed for a particular data point. The number of rows in
            // the matrix is the number of data points observed.
            std::unordered_map<int, Eigen::MatrixXd> data_per_time_slice;

            // True if the node represents a segment node.
            bool segment = false;

            // Set of incoming nodes that must be ignored when producing a
            // backward message to a certain target (the key of the mapping);
            std::unordered_map<std::string, std::unordered_set<std::string>>
                backward_blocking;

            // For prediction, we might want to transform this node distribution
            // into a binary distribution. The value here indicates the
            // assignment that represents the value 1 in the binary
            // distribution.
            int aggregation_value = -1;
        };

    } // namespace model
} // namespace tomcat
