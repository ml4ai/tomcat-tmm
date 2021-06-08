#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <fstream>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graphviz.hpp>

#include "pgm/RandomVariableNode.h"
#include "utils/Definitions.h"

namespace tomcat {
    namespace model {

        //------------------------------------------------------------------
        // Structs
        //------------------------------------------------------------------
        struct VertexData {
            // This needs to be a shared pointer because some of the nodes can
            // be parameter nodes sharable among some CPDs
            std::shared_ptr<RandomVariableNode> node;

            // Node timed name here just for visualization purposes.
            std::string label;
        };

        /**
         * Represents a Dynamic Bayes Net as a Directed Acyclic Graph. It is
         * comprised of node templates that are replicated into concrete timed
         * node instances when the DBN is unrolled. In this process, the nodes's
         * CPDs that depend on other nodes in the graph are updated to reference
         * the concrete instances of the nodes they depend on. By doing this,
         * it's guaranteed that a sample from these CPDs will condition on the
         * current assignments of the nodes in the DBN.
         */
        class DynamicBayesNet {
          public:
            //------------------------------------------------------------------
            // Types, Enums & Constants
            //------------------------------------------------------------------
            typedef std::pair<std::shared_ptr<RandomVariableNode>,
                              std::shared_ptr<RandomVariableNode>>
                Edge;

            //------------------------------------------------------------------
            // Constructors & Destructor
            //------------------------------------------------------------------

            /**
             * Creates an instance of a DBN.
             */
            DynamicBayesNet();

            /**
             * Creates an instance of a DBN and reserves space in the vector of
             * node templates.
             */
            DynamicBayesNet(int num_node_templates);

            ~DynamicBayesNet();

            //------------------------------------------------------------------
            // Copy & Move constructors/assignments
            //------------------------------------------------------------------
            DynamicBayesNet(const DynamicBayesNet&) = default;

            DynamicBayesNet& operator=(const DynamicBayesNet&) = default;

            DynamicBayesNet(DynamicBayesNet&&) = default;

            DynamicBayesNet& operator=(DynamicBayesNet&&) = default;

            //------------------------------------------------------------------
            // Static functions
            //------------------------------------------------------------------

            /**
             * Creates a model from nodes and connections defined in a json
             * file.
             *
             * @param filepath: location where the json file is
             *
             * @return DBN
             */
            static DynamicBayesNet
            create_from_json(const std::string& filepath);

            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            /**
             * Adds node to the DBN as a template. This function adds a deep
             * copy of the node to the list of templates by cloning the node so
             * that CPDs and nodes they depend on are also cloned. After
             * unrolling the DBN some timed instance nodes may share the same
             * CPD with some of the node templates so it's necessary to have
             * exclusive copies of these objects in the DBN to avoid conflict
             * with other processes.
             *
             * @param node: node to be stored in the DBN as a template
             */
            void
            add_node_template(const std::shared_ptr<RandomVariableNode>& node);

            /**
             * Unrolls the DBN into time steps if nor previously unrolled into
             * the same number of time steps. This process creates vertices and
             * edges in the underlying graph structure by replicating nodes over
             * time, storing them into vertices linked to each other according
             * to the definitions in the nodes' metadata. It also updates the
             * CPDs that depend on other nodes with the concrete timed instances
             * of such nodes.
             *
             * @param time_steps: number of time steps to unroll into
             * @param force: whether the DBN should be forced to unroll again
             * even if it was previously unrolled over the same number of time
             * steps. This is useful if the DBN needs to be unrolled in the same
             * number of time steps as before but changed somehow (e.g. more
             * nodes were added to it).
             */
            void unroll(int time_steps, bool force);

            /**
             * Expand the DBN by unrolling it into more time steps.
             *
             * @param new_time_steps: number of time steps to add to the
             * unrolled DBN
             */
            void expand(int new_time_steps);

            /**
             * Checks if the DBN is consistent and prepared to be unrolled.
             */
            void check();

            /**
             * Returns the list of timed-instance parameter nodes in the DBN.
             *
             * @return Parameter nodes.
             */
            std::vector<std::shared_ptr<Node>> get_parameter_nodes();

            /**
             * Returns the list of timed instance nodes created from the
             * template with with a specific label.
             *
             * @param label: node's label
             * @return
             */
            std::vector<std::shared_ptr<Node>>
            get_nodes_by_label(const std::string& node_label) const;

            /**
             * Returns timed node objects in topological order.
             *
             * @return Times node objects in topological order.
             */
            std::vector<std::shared_ptr<Node>>
            get_nodes_topological_order(bool from_roots_to_leaves = true) const;

            /**
             * Returns timed instances of the parents of a node
             * @param node: timed instance of a node
             * @param exclude_parameters: whether parameter nodes should be
             * excluded from the list of parent nodes
             * @return Time instances of a node's parents.
             */
            std::vector<std::shared_ptr<Node>>
            get_parent_nodes_of(const std::shared_ptr<Node>& node,
                                bool exclude_parameters) const;

            /**
             * Returns timed instances of the children of a node
             *
             * @param node: timed instance of a node
             * @param exclude_timers: exclude any child that is a timer node
             *
             * @return Time instances of a node's children.
             */
            std::vector<std::shared_ptr<Node>>
            get_child_nodes_of(const std::shared_ptr<Node>& node,
                               bool exclude_timers = false) const;

            /**
             * Saves model's parameter values in individual files inside a given
             * directory. The directory is created if it does not exist.
             *
             * @param output_dir: folder where the files must be saved
             */
            void save_to(const std::string& output_dir) const;

            /**
             * Loads model's parameter assignments from files previously saved
             * in a specific directory and freeze the parameter nodes.
             *
             * @param input_dir: directory where the files with the parameters'
             * values are saved
             * @param freeze_nodes: if true nodes that had their assignments
             * set will be frozen
             */
            void load_from(const std::string& input_dir, bool freeze_nodes);

            /**
             * Returns edges of the unrolled DBN
             *
             * @return Edges
             */
            std::vector<Edge> get_edges() const;

            /**
             * Returns the cardinality of any node derived from a given label.
             *
             * @param node_label: template node's label
             *
             * @return Cardinality of the template node.
             */
            int get_cardinality_of(const std::string& node_label) const;

            /**
             * Checks if the models has any node with a given label.
             *
             * @param node_label: node's label
             *
             * @return True if a node with the given label is part of the
             * model.
             */
            bool has_node_with_label(const std::string& node_label) const;

            /**
             * Checks if the models has any parameter node with a given label.
             *
             * @param node_label: node's label
             *
             * @return True if a node with any parameter node with a given
             * label is part of the model.
             */
            bool
            has_parameter_node_with_label(const std::string& node_label) const;

            /**
             * Return the labels of the parameter nodes in the DBN.
             *
             * @return Parameter nodes' labels.
             */
            std::vector<std::string> get_parameter_node_labels() const;

            /**
             * Gets a pointer to a node by its label and time step.
             *
             * @param label: node's label
             * @param time_step: node's time step
             *
             * @return Pointer to a node
             */
            RVNodePtr get_node(const std::string& label, int time_step);

            /**
             * Returns the metadata of any node derived from a given label.
             *
             * @param node_label: template node's label
             *
             * @return Metadata of the template node.
             */
            std::shared_ptr<NodeMetadata>
            get_metadata_of(const std::string& node_label) const;

            /**
             * Creates a deep copy of the template nodes and unrolls the DBN
             * into the same time steps as the original one. Only the
             * parameter nodes' assignments are preserved.
             *
             * @param unroll: whether the new DBN must be unrolled into the
             * same number of time steps as the original one.
             *
             * @return a deep copy of this DBN
             */
            DynamicBayesNet clone(bool unroll) const;

            /**
             * Set relevant attributes (assignment and frozen) of parameter
             * nodes in this DBN from a list of parameters of another DBN.
             *
             * @param dbn: DBN to cpy parameters from
             */
            void mirror_parameter_nodes_from(const DynamicBayesNet& dbn);

            /**
             * Retrieves the list of nodes in topological order (from the roots
             * to the leaves) over time.
             *
             * @param List of nodes.
             */
            RVNodePtrVec get_nodes_in_topological_order();

            /**
             * Retrieves the list of nodes in topological order (from the roots
             * to the leaves) in a specific time slice.
             *
             * @param List of nodes.
             */
            RVNodePtrVec get_nodes_in_topological_order_at(int time_step);

            /**
             * Writes the graph content in graphviz format.
             *
             * @param output_stream: output stream to write the graph.
             */
            void print_graph(std::ostream& output_stream) const;

            /**
             * Print CPDs of the template nodes in the model.
             *
             * @param output_stream: output stream to write the CPDs.
             */
            void print_cpds(std::ostream& output_stream) const;

            /**
             * Get a list of concrete nodes at a given time step.
             *
             * @param time_step: time step
             *
             * @return List of nodes
             */
            RVNodePtrVec get_data_nodes(int time_step) const;

            /**
             * Retrieves the list of data nodes (non parameter) in topological
             * order (from the roots to the leaves) in a specific time slice.
             *
             * @param List of data nodes.
             */
            RVNodePtrVec get_data_nodes_in_topological_order_at(int time_step);

            // --------------------------------------------------------
            // Getters & Setters
            // --------------------------------------------------------
            int get_time_steps() const;

            NodePtrVec get_nodes() const;

            const RVNodePtrVec& get_single_time_nodes() const;

          private:
            //------------------------------------------------------------------
            // Types, Enums & Constants
            //------------------------------------------------------------------

            // The graph is defined as bidirectional to speed up the access to
            // the list of parents and children of a vertex. However, only
            // single-direction edges will be created in reality.
            typedef boost::adjacency_list<boost::vecS,
                                          boost::vecS,
                                          boost::bidirectionalS,
                                          VertexData>
                Graph;

            typedef std::unordered_map<std::string, int> IDMap;

            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            /**
             * Clears graph and mappings. Only the list of node templates are
             * preserved.
             */
            void reset();

            /**
             * Creates vertices from a list of node templates.
             *
             * @param new_time_steps: number of replicable node copies over
             * time to add to the unrolled DBN
             */
            void create_vertices_from_nodes(int new_time_steps);

            /**
             * Creates a vertex in the graph and stores a node timed instance in
             * it.
             *
             * @param node_template: node template
             * @param time_step
             * @return
             */
            VertexData
            add_vertex(const std::shared_ptr<RandomVariableNode>& node_template,
                       int time_step);

            /**
             * Uses the node templates' metadata to link the vertices
             * accordingly.
             *
             * @param new_time_steps: number of replicable node copies over
             * time to add to the unrolled DBN
             */
            void create_edges(int new_time_steps);

            /**
             * Adds a new edge to the graph.
             *
             * @param source_node: node where the edge should start from
             * @param target_node: node where the edge should end in
             * @param time_crossing: whether the edge should cross between one
             * time step to the subsequent one
             * @param target_time_step: time step of the timed instance of the
             * target node template
             */
            void add_edge(const NodeMetadata& source_node_metadata,
                          const NodeMetadata& target_node_metadata,
                          bool time_crossing,
                          int target_time_step);

            /**
             * For each one of the nodes, this method sets its parents,
             * children and concrete CPD from the list of possible CPDs in
             * the node's metadata.
             */
            void set_parents_children_and_cpd_to_nodes();

            /**
             * Add the timed copy of a timer node to the node controlled by it.
             *
             * @param num_timed_copies: number of replicable node copies over
             * time to add to the unrolled DBN
             */
            void set_timers_to_nodes(int num_timed_copies);

            /**
             * Set the vector of timed copies to each repeatable node in the
             * DBN.
             *
             * @param num_timed_copies: number of replicable node copies over
             * time to add to the unrolled DBN
             */
            void set_timed_copies_to_nodes(int num_timed_copies);

            /**
             * Replaces node objects in the CPDs that depend on other nodes with
             * their concrete timed instance replica in the unrolled DBN.
             *
             * @param num_timed_copies: number of replicable node copies over
             * time to add to the unrolled DBN
             */
            void update_cpd_templates_dependencies(int num_timed_copies);

            /**
             * Save list of nodes in topological order per time step;
             */
            void save_topological_list();

            //------------------------------------------------------------------
            // Data members
            //------------------------------------------------------------------
            Graph graph;

            IDMap name_to_id;

            // List of concrete timed instances node of the unrolled DBN.
            NodePtrVec nodes;

            // List of concrete nodes per time step
            std::vector<RVNodePtrVec> data_nodes_per_time_step;

            RVNodePtrVec single_time_nodes;

            // List of nodes per time step and topological order.
            std::vector<RVNodePtrVec> topological_nodes_per_time;

            std::vector<RVNodePtrVec> topological_data_nodes_per_time;

            // Mapping between a timed instance parameter node's label and its
            // node object.
            Node::NodeMap parameter_nodes_map;

            // Mapping between a node label and all of the timed instance nodes
            // created from the template with such label.
            std::unordered_map<std::string, NodePtrVec> label_to_nodes;

            // Node templates will be used to create concrete instances of
            // nodes over time (timed node instances/objects), which will be
            // stored in the vertices of the unrolled DBN.
            //
            // The original list is preserved to allow multiple calls of the
            // unrolled method based on the original set of nodes.
            // TODO - change to a set to forbid adding the same node multiple
            //  times
            RVNodePtrVec node_templates;

            // If unrolled, the number of time steps the DBN was unrolled into
            int time_steps = 0;
        };

    } // namespace model
} // namespace tomcat
