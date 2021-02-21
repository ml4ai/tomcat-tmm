#pragma once

#include "Node.h"

#include "pgm/cpd/CPD.h"
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

        /**
         * A random variable node is a concrete node in the unrolled DBN
         * that has a distribution from which it can be sampled from. The
         * assignment of this node can change as we sample from it's posterior
         * distribution over the other nodes' assignments in the unrolled DBN.
         */
        class RandomVariableNode
            : public Node,
              public std::enable_shared_from_this<RandomVariableNode> {
          public:
            //------------------------------------------------------------------
            // Constructors & Destructor
            //------------------------------------------------------------------

            /**
             * Creates an instance of a random variable node. A random variable
             * node has CPDs associated with it and its assigned value is
             * mutable.
             *
             * @param metadata: node's metadata
             * @param time_step: node's time step in an unrolled DBN (0 by
             * default)
             */
            RandomVariableNode(const std::shared_ptr<NodeMetadata>& metadata,
                               int time_step = 0);

            /**
             * Creates an instance of a random variable node. A random variable
             * node has CPDs associated with it and its assigned value is
             * mutable.
             *
             * @param metadata: node's metadata
             * @param time_step: node's time step in an unrolled DBN (0 by
             * default)
             */
            RandomVariableNode(std::shared_ptr<NodeMetadata>&& metadata,
                               int time_step = 0);

            ~RandomVariableNode();

            //------------------------------------------------------------------
            // Copy & Move constructors/assignments
            //------------------------------------------------------------------
            RandomVariableNode(const RandomVariableNode& node);

            RandomVariableNode& operator=(const RandomVariableNode& node);

            RandomVariableNode(RandomVariableNode&&) = default;

            RandomVariableNode& operator=(RandomVariableNode&&) = default;

            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------
            std::string get_description() const override;

            std::unique_ptr<Node> clone() const override;

            std::string get_timed_name() const override;

            /**
             * Replaces parameter nodes in node dependent CPD templates by a
             * concrete timed-instance node in the unrolled DBN.
             *
             * @param parameter_nodes_map: mapping between a parameter node
             * timed name and its object in an unrolled DBN
             */
            void update_cpd_templates_dependencies(
                const NodeMap& parameter_nodes_map);

            /**
             * Create new references for the CPD templates of the node.
             */
            void clone_cpd_templates();

            /**
             * Generates samples from this node's CPD given its parents'
             * assignments.
             *
             * @param random_generator_per_job: random number generator per
             * thread
             * @param num_samples: number of samples to generate
             *
             * @return Samples from the node's CPD.
             */
            Eigen::MatrixXd sample(const std::vector<std::shared_ptr<gsl_rng>>&
                                       random_generator_per_job,
                                   int num_samples) const;

            /**
             * Returns p(children(node)|node). The posterior of a node is
             * given by p(node|parents(node)) * p(children(node)|node). We
             * call the second term, the posterior weights here.
             *
             * Note: This method changes this node assignment temporarily while
             * calculating the weights, therefore it's not const. The final
             * state of the this object is unchanged though.
             *
             * @param num_jobs: number of threads to perform vertical
             * parallelization (split the computation over the
             * observations/data points provided). If 1, the computations are
             * performed in the main thread
             * @param min_time_step_to_sample: get cached posterior weights for
             * children from time step smaller than this value
             * @param max_time_step_to_sample: ignore children from time step
             * larger than this value
             * @param use_weights_cache: whether the node must use cache to process
             * posterior weights os multi-time nodes when the samples are
             * generated to do forward inference.
             *
             * @return Posterior weights
             */
            Eigen::MatrixXd get_posterior_weights(int num_jobs,
                                                  int min_time_step_to_sample,
                                                  int max_time_step_to_sample,
                                                  bool use_weights_cache);

            /**
             * Samples a node using conjugacy properties and sufficient
             * statistics stored in the node's CPD.
             *
             * @param random_generator: random number generator
             * @param num_samples: number of samples to generate
             * @return
             */
            Eigen::MatrixXd sample_from_conjugacy(
                const std::shared_ptr<gsl_rng>& random_generator,
                int num_samples) const;

            /**
             * Update sufficient statistics of parent parameter nodes with this
             * node's assignment(s).
             *
             */
            void update_parents_sufficient_statistics();

            /**
             * Adds a set of values to the sufficient statistics of a
             * parameter node's CPD.
             *
             * @param values: Values to add to the sufficient statistics. The
             * update_parents_sufficient_statistics will call this function for
             * parameter nodes at some point.
             */
            void
            add_to_sufficient_statistics(const std::vector<double>& values);

            /**
             * Clears the values stored as sufficient statistics in the
             * parameter node CPD.
             */
            void reset_sufficient_statistics();

            /**
             * Prevents node's assignment to be changed.
             */
            void freeze();

            /**
             * Frees node's assignment to be changed.
             */
            void unfreeze();

            /**
             * Adds a CPD to the list of possible CPDs of the node.
             *
             * @param cpd: CPD
             */
            void add_cpd_template(const std::shared_ptr<CPD>& cpd);

            /**
             * Adds CPD to the list of possible CPDs of the node.
             *
             * @param cpd: CPD
             */
            void add_cpd_template(std::shared_ptr<CPD>&& cpd);

            /**
             * Checks if a replicable node that depends on a past copy of
             * itself is the first to index a CPD. We require this to speed
             * up some computations when using categorical CPDs.
             *
             * @param cpd: cpd
             */
            void check_cpd(const std::shared_ptr<CPD>& cpd) const;

            /**
             * Returns the node's CPD associated to a set of parents.
             *
             * @param parent_labels: labels of the parents of the node
             *
             * @return node's CPD related to the parents informed
             */
            std::shared_ptr<CPD>
            get_cpd_for(const std::vector<std::string>& parent_labels) const;

            /**
             * Checks whether the node has a timer associated with it.
             *
             * @return True if the node is controlled by a timer.
             */
            bool has_timer() const;

            /**
             * Retrieves copy of the node increment time steps behind.
             *
             * @param increment: number of time steps to jump in the past
             *
             * @return Previous copy in time.
             */
            std::shared_ptr<RandomVariableNode>
            get_previous(int increment = 1) const;

            /**
             * Retrieves copy of the node increment time steps ahead.
             *
             * @param increment: number of time steps to jump in the future
             *
             * @return Next copy in time.
             */
            std::shared_ptr<RandomVariableNode>
            get_next(int increment = 1) const;

            /**
             * Indicates whether the node follows a continuous distribution
             * or not.
             *
             * @return Whether the node follows a continuous distribution
             * or not.
             */
            bool is_continuous() const;

            /**
             * Resets saved data used to speed up the sampling process.
             */
            void clear_cache();

            // -----------------------------------------------------------------
            // Virtual functions
            // -----------------------------------------------------------------

            /**
             * Generates a sample for the node from its posterior distribution.
             * If the node is an in-plate node, multiple samples will be
             * generated for the node (one per row). The number of total
             * samples are defined by the number of elements in-plate.
             *
             * Note: This method changes this node assignment temporarily while
             * calculating the weights, therefore it's not const. The final
             * state of the this object is unchanged though.
             *
             * @param random_generator_per_job: random number generator per
             * thread
             * @param min_time_step_to_sample: min time step to consider when
             * computing the posterior distribution to sample from. This
             * information is used to cache previously computed posterior
             * weights for the node's children when estimation is being done.
             * @param max_time_step_to_sample: max time step to consider when
             * computing the posterior distribution to sample from. The DBN
             * might have more time steps but we want to ignore time steps
             * larger than the max_time_step_to_sample. This means that
             * children or timers in a time step bigger than this attribute
             * won't be considered in the computation.
             * @param use_cache: whether the node must use cache to process
             * posterior weights os multi-time nodes when the samples are
             * generated to do forward inference.
             *
             * @return Sample for the node from its posterior
             */
            virtual Eigen::MatrixXd
            sample_from_posterior(const std::vector<std::shared_ptr<gsl_rng>>&
                                      random_generator_per_job,
                                  int min_time_step_to_sample,
                                  int max_time_step_to_sample,
                                  bool use_weights_cache);

            // -----------------------------------------------------------------
            // Getters & Setters
            // -----------------------------------------------------------------
            int get_time_step() const;

            void set_time_step(int time_step);

            void set_assignment(const Eigen::MatrixXd& assignment);

            bool is_frozen() const;

            const std::shared_ptr<CPD>& get_cpd() const;

            void set_cpd(const std::shared_ptr<CPD>& cpd);

            const std::vector<std::shared_ptr<Node>>& get_parents() const;

            const std::vector<std::shared_ptr<Node>>& get_children() const;

            void set_parents(const std::vector<std::shared_ptr<Node>>& parents);

            void
            set_children(const std::vector<std::shared_ptr<Node>>& children);

            const std::shared_ptr<TimerNode>& get_timer() const;

            void set_timer(const std::shared_ptr<TimerNode>& timer);

            void
            set_timed_copies(const std::shared_ptr<
                             std::vector<std::shared_ptr<RandomVariableNode>>>&
                                 timed_copies);

          protected:
            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            /**
             * Copies data members of a random variable node.
             *
             * @param cpd: continuous CPD
             */
            void copy_node(const RandomVariableNode& node);

            /**
             * Create new reference for the CPD of the node.
             */
            void clone_cpd();

            /**
             * Sorts a list of labels and concatenate them into a string
             * separated by a delimiter.
             *
             * @param labels: list of random variable node's labels *
             * @return Unique string formed by a list of random variable node's
             * labels
             */
            std::string get_unique_key_from_labels(
                const std::vector<std::string>& labels) const;

            // -----------------------------------------------------------------
            // Data members
            // -----------------------------------------------------------------

            // Time step where the node shows up in the unrolled DBN. This
            // variable will be assigned when a concrete timed instance of this
            // node is created and assigned to a vertex in an unrolled DBN.
            int time_step = 0;

            // CPD is a shared pointer because multiple nodes can have a CPD
            // that depend on the same set of parameters.
            // A node can also have more than one CPD, each one referring to a
            // conditional distribution given a set of different parents. For
            // instance, a State node in a HMM will likely have two CPDs: one
            // for the first time step with no parent nodes associated and
            // another one to the following time steps with a State from the
            // previous time step as a parent.
            // The key in this data structure is a string that uniquely identify
            // the set of parents of the node to which the CPD is related.
            std::unordered_map<std::string, std::shared_ptr<CPD>> cpd_templates;

            // Conditional probability distribution of the timed instance of the
            // node in the unrolled DBN. The node template contains a list of
            // possible CPDs that can be associated with it. Once a concrete
            // timed instance of the node is created in the unrolled DBN, the
            // set of parents of such node is known and its CPD can be fully
            // determined.
            std::shared_ptr<CPD> cpd;

            /**
             * A frozen node will have it's assignment preserved and won't be
             * considered a latent node by the samplers. It will behave like a
             * constant node.
             */
            bool frozen = false;

            NodePtrVec parents;

            std::vector<NodePtrVec> children_per_time_step;

            // If set, the amount of time this node stays in the current
            // state, is defined by the timer's assignment (semi-Markov model).
            std::shared_ptr<TimerNode> timer;

            // Vector of instance of the current node in each one of the time
            // steps from its initial appearance until the last one. This
            // allows us easy access to any other instance in time.
            std::shared_ptr<std::vector<std::shared_ptr<RandomVariableNode>>>
                timed_copies;

          private:
            //------------------------------------------------------------------
            // Structs
            //------------------------------------------------------------------
            struct PosteriorWeightsCache {
                Eigen::MatrixXd cum_log_weights_from_children;
                Eigen::MatrixXd log_weights_from_children;
                int time_step = 0;
            };

            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            /**
             * Gets the collection of children of this node within a range
             * of time steps.
             *
             * @param min_time_step: lower bound
             * @param max_time_step: upper bound (inclusive)
             * @return
             */
            NodePtrVec get_children_in_range(int min_time_step,
                                             int max_time_step);

            /**
             * Computes posterior weights from the left, central and right
             * segments from a time controlled node.
             *
             *  1. If node == left seg. values and node == right seg. values
             *  p(duration left + 1 + duration right)
             *
             *  2. If node == left seg. values and node != right seg. values
             *  p(duration left + 1)p(right seg. value | node value)
             *  p(duration right)
             *
             *  3. If node != left seg. values and node == right seg. values
             *  p(duration left)p(left seg. value | node)
             *  p(duration right + 1)
             *
             *  4. If node != left seg. values and node != right seg. values
             *  p(duration left)p(left seg. value | node)
             *  p(duration central == 1)p(right seg. value | node value)
             *  p(duration right)
             *
             * @param num_jobs: number of threads to perform vertical
             * parallelization (split the computation over the
             * observations/data points provided). If 1, the computations are
             * performed in the main thread
             * @param max_time_step_to_sample: clamp the right segment to
             * this value
             *
             * @return Posterior weights for the left, central and right
             * segments combined
             */
            Eigen::MatrixXd
            get_segments_log_posterior_weights(int num_jobs,
                                               int max_time_step_to_sample);

            /**
             * Accumulate cached weights from children computed from the immediate
             * previous sampling range.
             *
             * @param min_time_step_to_sample: min time step to sample in the
             * current sampling range
             *
             */
            void accumulate_cached_log_weights(int
            min_time_step_to_sample);

            /**
             * Gets the cached weights from children computed from the immediate
             * previous sampling range.
             *
             * @param min_time_step_to_sample: min time step to sample in the
             * current sampling range
             *
             * @return Pre-computed log posterior weights
             */
            Eigen::MatrixXd get_cached_log_weights(int
            min_time_step_to_sample) const;

            /**
             * Save log weights of relevant children in the cache to be
             * accumulated as the estimation process advances to the next
             * sampling range.
             *
             * @param min_time_step_to_sample: min time step to sample in the
             * current sampling range
             * @param child_node: child node which weights were computed for
             * @param log_weights: log weights computed for the child
             */
            void cache_current_log_weights(int min_time_step_to_sample,
                                   const RVNodePtr& child_node,
                                   const Eigen::MatrixXd& log_weights);

            //------------------------------------------------------------------
            // Data members
            //------------------------------------------------------------------

            // Stores posterior weights computed for children in time steps
            // previous to the minimum time step to which posteriors are
            // being computed. This speed up computation when we are sampling
            // from posterior for estimation purposes.
            PosteriorWeightsCache cached_posterior_weights;
        };

    } // namespace model
} // namespace tomcat
