#pragma once

#include "sampling/Sampler.h"

#include <memory>
#include <mutex>

#include "AncestralSampler.h"
#include "utils/Definitions.h"

namespace tomcat {
    namespace model {

        /**
         * Generates samples by Gibbs sampling process. First, samples are
         * generated for data nodes (non-parameter nodes) from the roots to the
         * leaves. Samples values from these nodes are propagated as sufficient
         * statistics to parameter nodes they depend on. Next, parameter nodes
         * are sampled from their posterior formed by it's conjugate prior
         * updated by the sufficient statistics previously propagated by the
         * data nodes.
         *
         */
        class GibbsSampler : public Sampler {
          public:
            //------------------------------------------------------------------
            // Constructors & Destructor
            //------------------------------------------------------------------

            /**
             * Creates an instance of the sampler for a given model and random
             * number generator.
             *
             * @param model: DBN
             * @param burn_in_period: number of passes before samples are stored
             * @param num_jobs: number of threads created for parallel
             * sampling. If 1, no parallel processing is performed and  the code
             * runs in the main thread.
             */
            GibbsSampler(const std::shared_ptr<DynamicBayesNet>& model,
                         int burn_in_period,
                         int num_jobs = 1);

            ~GibbsSampler();

            //------------------------------------------------------------------
            // Copy & Move constructors/assignments
            //------------------------------------------------------------------
            GibbsSampler(const GibbsSampler& sampler);

            GibbsSampler& operator=(const GibbsSampler& sampler);

            GibbsSampler(GibbsSampler&&) = default;

            GibbsSampler& operator=(GibbsSampler&&) = default;

            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------
            void sample_latent(const std::shared_ptr<gsl_rng>& random_generator,
                               int num_samples) override;

            Tensor3 get_samples(const std::string& node_label) const override;

            Tensor3 get_samples(const std::string& node_label,
                                int low_time_step,
                                int high_time_step) const override;

            void get_info(nlohmann::json& json) const override;

            void prepare() override;

          protected:
            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------
            void copy_sampler(const GibbsSampler& sampler);

          private:
            //------------------------------------------------------------------
            // Structs
            //------------------------------------------------------------------
            struct NodeSet {
                /**
                 * Struct that contains a collection of nodes separate in
                 * distinct lists to speed up processing.
                 */
                NodePtrVec sampled_nodes;
                NodePtrVec timer_nodes;

                // The nodes in this list cannot be processed in parallel
                // over time because they can either depend on nodes across
                // all the time steps, or they can be a timer or controlled by
                // one, in which case, their dependencies over time are
                // random. The computation in these nodes will be
                // parallelized over the data provided (vertical
                // parallelization).
                NodePtrVec single_thread_over_time_nodes;

                // Parameter nodes can all be sampled in parallel since they
                // are sampled from conjugacy
                std::vector<NodePtrVec> parameter_nodes_per_job;

                // We divide data nodes in two categories: data nodes at even
                // and odd time steps. This guarantees that nodes in each of
                // these categories can be sampled in parallel as nodes from a
                // time step can only depend on nodes from previous, current or
                // subsequent time steps.
                std::vector<NodePtrVec> even_time_data_nodes_per_job;
                std::vector<NodePtrVec> odd_time_data_nodes_per_job;
            };

            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            /**
             * Gets a collection of nodes split into lists for fast
             * processing.
             *
             * @return Node set
             */
            NodeSet get_node_set() const;

            /**
             * Populates the nodes' assignments with initial values.
             *
             * @param random_generator: random number generator
             */
            void fill_initial_samples(
                const std::shared_ptr<gsl_rng>& random_generator);

            /**
             * Initialize tensors to store the samples generated by the latent
             * nodes.
             *
             * @param num_samples: number of samples to generate
             * @param latent_nodes: nodes which samples will be generated for
             */
            void init_samples_storage(
                int num_samples,
                const std::vector<std::shared_ptr<Node>>& latent_nodes);

            /**
             * Initialize timer nodes by doing one pass forward and backwards;
             *
             * @param node_set: collection of nodes to be processed
             */
            void
            init_timers(const std::vector<std::shared_ptr<Node>> timer_nodes);

            /**
             * Sample a collection of independent nodes in parallel.
             *
             * @param random_generators_per_job: random number generators to
             * use in each job
             * @param nodes_per_job: list of data nodes to process in each job
             * @param discard: whether samples generated must be discarded or
             * not
             * @param data_nodes: whether the nodes are data nodes or
             * parameter notes otherwise
             */
            void sample_nodes_in_parallel(
                const std::vector<std::shared_ptr<gsl_rng>>&
                    random_generators_per_job,
                const std::vector<std::vector<std::shared_ptr<Node>>>&
                    nodes_per_job,
                bool discard,
                bool data_nodes);

            /**
             * Samples a series of nodes from their posterior distribution.
             *
             * @param random_generator: random number generator
             * @param nodes: nodes to be processed
             * @param discard: indicates whether the samples should be discarded
             * or stored
             */
            void run_sample_from_posterior_thread(
                const std::shared_ptr<gsl_rng>& random_generator,
                const std::vector<std::shared_ptr<Node>>& nodes,
                bool discard);

            /**
             * Samples from the posterior distribution of a node. The posterior
             * for a data node x is given by
             * P(x|Pa(x))P(Ch1(x)|x)...P(Chk(x)|x), where Pa(x) is the parents
             * of x and Chi(x) is the i-th child of x. P(Chi(x)|x) is a vector
             * containing the pdf of the current value assigned to the i-th
             * child f x for each one of the possible values x can take. For
             * parameter nodes, the posterior is sampled from prior sufficient
             * statistics updating via conjugacy.
             *
             * @param random_generator_per_job: random number generator per
             * thread
             * @param node: node
             * @param discard: indicates whether the sample should be discarded
             * or stored
             * @param update_sufficient_statistics: indicates whether
             * sufficient statistics of the node's CPD's parameter's priors
             * must be updated after generating a sample from the node
             * (relevant for data nodes)
             */
            void
            sample_from_posterior(const std::vector<std::shared_ptr<gsl_rng>>&
                                      random_generator_per_job,
                                  const std::shared_ptr<Node>& node,
                                  bool discard,
                                  bool update_sufficient_statistics = true);

            /**
             * Sample nodes that cannot be horizontally parallelized (timer
             * and time controlled nodes).
             *
             * @param random_generator_per_job: random number generator per
             * thread
             * @param single_thread_nodes: nodes that cannot be horizontally
             * split in multiple threads.
             * @param discard: whether the sampled must be saved or discarded
             */
            void sample_single_thread_nodes(
                const std::vector<std::shared_ptr<gsl_rng>>&
                    random_generator_per_job,
                const std::vector<std::shared_ptr<Node>>& single_thread_nodes,
                bool discard);

            /**
             * Update sufficient statistics of the timer nodes.
             *
             * @param timer_nodes: timer nodes
             */
            void update_timer_sufficient_statistics(
                const std::vector<std::shared_ptr<Node>>& timer_nodes);

            /**
             * Stores a new sample in the local samples' container.
             *
             * @param node: Node which the sample was generated for
             * @param sample: Sampled value
             */
            void keep_sample(const std::shared_ptr<RandomVariableNode>& node,
                             const Eigen::MatrixXd& sample);

            /**
             * Reset samples counter and storage
             */
            void reset();

            //------------------------------------------------------------------
            // Data members
            //------------------------------------------------------------------
            int burn_in_period = 0;

            // Container to store the samples generated by this sampler.
            // Different from the Ancestral sampler, where the generated samples
            // are store as nodes' assignments, in the Gibbs sampler, for each
            // latent node, one sample is generate at a time and kept as that
            // node's assignment until the next round. Another structure is then
            // needed to stored the samples that were generated along the
            // process.
            std::unordered_map<std::string, Tensor3> node_label_to_samples;

            int iteration = 0;

            // Mutex to handle race condition when multiple threads try to
            // store samples at the same time.
            std::unique_ptr<std::mutex> keep_sample_mutex;

            void print_nodes(const NodeSet& node_set) const;

            // Selection of nodes used by the sampler. We store this
            // information to avoid computing this list every time tha sample
            // function is called.
            NodeSet node_set;
        };

    } // namespace model
} // namespace tomcat
