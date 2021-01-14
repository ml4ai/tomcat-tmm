#pragma once

#include <iterator>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <eigen3/Eigen/Dense>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>

#include "distribution/Distribution.h"
#include "pgm/Node.h"
#include "utils/Definitions.h"

namespace tomcat {
    namespace model {

        //------------------------------------------------------------------
        // Forward declarations
        //------------------------------------------------------------------

        class RandomVariableNode;
        class TimerNode;

        //------------------------------------------------------------------
        // Structs
        //------------------------------------------------------------------

        /** This struct stores indexing information for fast
         * retrieval of a distribution in the CPD's list of distributions given
         * parent nodes' assignments.
         */
        struct ParentIndexing {
            // Order of the parent node's label for table indexing
            int order;
            int cardinality;
            // Cumulative cardinality of the nodes to the right of the parent
            // node's label order.
            int right_cumulative_cardinality;

            ParentIndexing() {}
            ParentIndexing(int order,
                           int cardinality,
                           int right_cumulative_cardinality)
                : order(order), cardinality(cardinality),
                  right_cumulative_cardinality(right_cumulative_cardinality) {}
        };

        /**
         * Abstract representation of a conditional probability distribution.
         *
         * A CPD is comprised of a list of distributions that correspond to the
         * distribution of a child node given its parents' assignments. A CPD
         * can also be comprised of a single distribution if the CPD defines a
         * prior (not conditioned on any parent node).
         */
        class CPD {
          public:
            //------------------------------------------------------------------
            // Types, Enums & Constants
            //------------------------------------------------------------------
            typedef std::unordered_map<std::string, ParentIndexing>
                TableOrderingMap;

            //------------------------------------------------------------------
            // Constructors & Destructor
            //------------------------------------------------------------------

            /**
             * Creates an abstract representation of a conditional probability
             * distribution.
             */
            CPD();

            /**
             * Creates an abstract representation of a conditional probability
             * distribution.
             *
             * @param parent_node_order: evaluation order of the parent
             */
            CPD(const std::vector<std::shared_ptr<NodeMetadata>>&
                    parent_node_order);

            /**
             * Creates an abstract representation of a conditional probability
             * distribution.
             *
             * @param parent_node_order: evaluation order of the parent
             */
            CPD(std::vector<std::shared_ptr<NodeMetadata>>&& parent_node_order);

            /**
             * Creates an abstract representation of a conditional probability
             * distribution.
             *
             * @param parent_node_order: evaluation order of the parent
             * @param distributions: distributions of the CPD
             * nodes' assignments for correct distribution indexing
             */
            CPD(const std::vector<std::shared_ptr<NodeMetadata>>&
                    parent_node_order,
                const std::vector<std::shared_ptr<Distribution>>&
                    distributions);

            /**
             * Creates an abstract representation of a conditional probability
             * distribution.
             *
             * @param parent_node_label_order: evaluation order of the parent
             * @param distributions: distributions of the CPD
             * nodes' assignments for correct distribution indexing
             */
            CPD(std::vector<std::shared_ptr<NodeMetadata>>&& parent_node_order,
                std::vector<std::shared_ptr<Distribution>>&& distributions);

            virtual ~CPD();

            //------------------------------------------------------------------
            // Copy & Move constructors/assignments
            //------------------------------------------------------------------

            // Copy constructor and assignment should be deleted to avoid
            // implicit slicing and loss of polymorphic behaviour in the
            // subclasses. To deep copy, the clone method must be used.
            CPD(const CPD&) = delete;

            CPD& operator=(const CPD&) = delete;

            CPD(CPD&& cpd) = default;

            CPD& operator=(CPD&& cpd) = default;

            //------------------------------------------------------------------
            // Operator overload
            //------------------------------------------------------------------
            friend std::ostream& operator<<(std::ostream& os, const CPD& cpd);

            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            /**
             * Replaces parameter nodes in a node dependent CPD by the correct
             * copy of the node in an unrolled DBN. When a CPD is defined,
             * random variable nodes can be assigned the parameters of the
             * distributions it contains. As values are assigned to these node,
             * they serves as concrete parameter values for the underlying
             * distribution associated with their corresponding nodes.
             *
             * @param parameter_nodes_map: mapping between a parameter node's
             * timed name and its concrete object reference in an unrolled DBN
             * @param time_step: time step of the node that owns the CPD in the
             * unrolled DBN. It can be different from the time step of the
             * parameter node if the latter is shared among nodes over several
             * time steps.
             */
            void update_dependencies(const Node::NodeMap& parameter_nodes_map,
                                     int time_step);

            /**
             * Draws a sample from the distribution associated with the parent
             * nodes' assignments.
             *
             * @param random_generator: random number random_generator
             * @param num_samples: number of samples to generate.
             * @param cpd_owner: node to which sample is being generated,
             * which is also the owner of this CPD.
             *
             * @return A sample from one of the distributions in the CPD.
             */
            Eigen::MatrixXd
            sample(const std::shared_ptr<gsl_rng>& random_generator,
                   int num_samples,
                   const std::shared_ptr<const RandomVariableNode>& cpd_owner)
                const;

            /**
             * Generates a sample for the node that owns this CPD from its
             * posterior distribution.
             *
             * @param random_generator: random number generator
             * @param posterior_weights: posterior weights given by the product
             * of p(children(node)|node)
             *
             * @return Sample from the node's posterior.
             */
            Eigen::MatrixXd sample_from_posterior(
                const std::shared_ptr<gsl_rng>& random_generator,
                const Eigen::MatrixXd& posterior_weights,
                const std::shared_ptr<const RandomVariableNode>& cpd_owner)
                const;

            /**
             * Returns the indices of the distributions indexed by the current
             * indexing nodes' assignments.
             *
             * @param index_nodes: concrete objects of the nodes used to
             * index the CPD
             * @param num_indices: number of assignments of the indexing
             * nodes to consider.
             *
             * @return Indices of the distributions indexed by the current
             * indexing nodes' assignments.
             */
            std::vector<int> get_indexed_distribution_indices(
                const std::vector<std::shared_ptr<Node>>& index_nodes,
                int num_indices) const;

            /**
             * Update the sufficient statistics of parameter nodes the cpd
             * depend on with assignments of the cpd's owner.
             *
             * @param cpd_owner: Node that owns this CPD
             */
            void update_sufficient_statistics(
                const std::shared_ptr<RandomVariableNode>& cpd_owner);

            /**
             * Marks the CPD as not updated to force dependency update on a
             * subsequent call to the member function update_dependencies.
             */
            void reset_updated_status();

            /**
             * Prints a short description of the distribution.
             *
             * @param os: output stream
             */
            void print(std::ostream& os) const;

            /**
             * Return the matrix formed by the concrete assignments of the nodes
             * it depends on.
             *
             * @param parameter_idx: index of the parameter's assignment to
             * consider
             *
             * @return CPD table
             */
            Eigen::MatrixXd get_table(int parameter_idx) const;

            /**
             * Returns p(cpd_owner_assignments | left and right segments).
             * This method is only used for semi-Markov models where some
             * node is controlled by a timer node.
             *
             * @param sampled_node: random variable for with the posterior is
             * being computed
             * @param cpd_owner: Node that owns this CPD
             *
             * @return Posterior weights of the node that owns this CPD given
             * its left and right segments.
             */
            Eigen::MatrixXd get_segment_posterior_weights(
                const std::shared_ptr<RandomVariableNode>& sampled_node,
                const std::shared_ptr<const TimerNode>& cpd_owner) const;

            /**
             * Calculates p(left_segment_duration|node, right_segment) for a
             * given row in the node's assignment.
             *
             * @param sample_idx: row in the node's assignment to consider
             * @param segment_first_timer: timer in the beginning of the left
             * segment of a node
             *
             * @return  p(left_segment_duration|node, right_segment)
             */
            Eigen::VectorXd get_left_segment_posterior_weights(
                int sample_idx,
                const std::shared_ptr<const TimerNode>& segment_first_timer)
            const;

            //------------------------------------------------------------------
            // Virtual functions
            //------------------------------------------------------------------

            /**
             * Returns p(cpd_owner_assignments | sampled_node)
             *
             * @param index_nodes: concrete objects of the nodes used to
             * index the CPD
             * @param sampled_node: random variable for with the posterior is
             * being computed
             * @param cpd_owner: Node that owns this CPD
             *
             * @return Posterior weights of the node that owns this CPD for one
             * of its parent nodes.
             */
            virtual Eigen::MatrixXd get_posterior_weights(
                const std::vector<std::shared_ptr<Node>>& index_nodes,
                const std::shared_ptr<RandomVariableNode>& sampled_node,
                const std::shared_ptr<const RandomVariableNode>& cpd_owner)
                const;

            //------------------------------------------------------------------
            // Pure virtual functions
            //------------------------------------------------------------------

            /**
             * Creates a new unique pointer from a concrete instance of a CPD.
             *
             * @return Pointer to the new CPD.
             */
            virtual std::unique_ptr<CPD> clone() const = 0;

            /**
             * Adds a set of values to the sufficient statistics of this CPD.
             *
             * @param sample: Sample to add to the sufficient statistics.
             */
            virtual void
            add_to_sufficient_statistics(const std::vector<double>& values) = 0;

            /**
             * Samples using conjugacy properties and sufficient statistics
             * stored in the CPD.
             *
             * @param random_generator: random number generator
             * @param num_samples: number of samples to generate
             * @param cpd_owner: owner of the CPD and also parameter node to
             * which samples will be generated
             * @return
             */
            virtual Eigen::MatrixXd sample_from_conjugacy(
                const std::shared_ptr<gsl_rng>& random_generator,
                int num_samples,
                const std::shared_ptr<const RandomVariableNode>& cpd_owner)
                const = 0;

            /**
             * Clear the values stored as sufficient statistics;
             */
            virtual void reset_sufficient_statistics() = 0;

            //------------------------------------------------------------------
            // Getters & Setters
            //------------------------------------------------------------------
            const std::string& get_id() const;

            bool is_updated() const;

            const TableOrderingMap& get_parent_label_to_indexing() const;

          protected:
            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            /**
             * Copies data members of a CPD.
             *
             * @param cpd: CPD
             */
            void copy_cpd(const CPD& cpd);

            //------------------------------------------------------------------
            // Pure virtual functions
            //------------------------------------------------------------------

            /**
             * Returns a short description of the CPD.
             *
             * @return CPD's description.
             */
            virtual std::string get_description() const = 0;

            /**
             * Clones the distributions (and the nodes associated to them) of
             * the CPD.
             */
            virtual void clone_distributions() = 0;

            //------------------------------------------------------------------
            // Data members
            //------------------------------------------------------------------

            // Unique identifier formed by the concatenation of the parent
            // labels in alphabetical order delimited by comma.
            std::string id;

            // It defines the order of the parent nodes in the cartesian
            // product of their possible assignments. It's necessary to know
            // this order for correctly index a distribution given parent's
            // assignments.
            std::vector<std::shared_ptr<NodeMetadata>> parent_node_order;

            // List of distributions per parents' assignments
            std::vector<std::shared_ptr<Distribution>> distributions;

            // It indicates whether the CPD was updated with concrete instances
            // of the nodes it depends on
            bool updated = false;

            // Maps an indexing node's label to its indexing struct.
            TableOrderingMap parent_label_to_indexing;

            // Controls racing conditions when multiple threads are trying to
            // update the sufficient statistics. I decided to encapsulate it
            // in a pointer to allow CPDs to have default move constructors
            // as a mutex object cannot be moved. Since no copy or move will
            // be performed by different threads (they just update the
            // sufficient statistics), there's no need to deal with race
            // condition when copying or moving an object of this class.
            std::unique_ptr<std::mutex> sufficient_statistics_mutex;

          private:
            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            /**
             * Fills CPD's unique identifier formed by the concatenation of the
             * index nodes' labels in alphabetical order delimited by comma.
             */
            void init_id();

            /**
             * Fills mapping table for quick access to a distribution in the CPD
             * table given parent node's assignments.
             */
            void fill_indexing_mapping();

            /**
             * Returns the distribution indexed by the combination of
             * assignments of the nodes that index this CPD for a given row
             * in their assignments.
             *
             * @param index_nodes: nodes that index this CPD
             * @param sample_idx: row of their assignments to consider
             *
             * @return Index of a distribution from this CPD
             */
            int get_indexed_distribution_idx(
                const std::vector<std::shared_ptr<Node>>& index_nodes,
                int sample_idx) const;
        };

    } // namespace model
} // namespace tomcat
