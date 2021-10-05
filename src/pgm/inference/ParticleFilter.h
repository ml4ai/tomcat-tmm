#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <gsl/gsl_rng.h>

#include "pgm/DynamicBayesNet.h"
#include "pgm/EvidenceSet.h"
#include "utils/Definitions.h"

#include "multithread/ThreadPool.h"
#include "utils/Multithreading.h"

namespace tomcat {
    namespace model {

        /**
         * Represents a particle filter process to generate samples forward in
         * time.
         */
        class ParticleFilter {
          public:
            //------------------------------------------------------------------
            // Constructors & Destructor
            //------------------------------------------------------------------

            /**
             * Creates a particle filter for a specific dynamic bayes net.
             *
             * @param dbn: dynamic bayes net
             * @param num_particles: number of particles to be generated
             * @param random_generator: random number generator
             * @param num_jobs: number of threads created for parallel
             * sampling. If 1, no parallel processing is performed and the code
             * runs in the main thread.
             */
            ParticleFilter(const DynamicBayesNet& dbn,
                           int num_particles,
                           const std::shared_ptr<gsl_rng>& random_generator,
                           int num_jobs = 1);

            ~ParticleFilter();

            //------------------------------------------------------------------
            // Copy & Move constructors/assignments
            //------------------------------------------------------------------

            ParticleFilter(const ParticleFilter&) = delete;

            ParticleFilter& operator=(const ParticleFilter&) = delete;

            ParticleFilter(ParticleFilter&&) = default;

            ParticleFilter& operator=(ParticleFilter&&) = default;

            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            /**
             * Generates particles for the next time steps given observations.
             * Particles are generated for the number of time steps contained in
             * the new_data. The last particles are stored in the nodes of the
             * template DBN. This function uses observations of a single data
             * point.
             *
             * @param new_data: evidence for the following time steps.
             *
             * @return Samples generated for each one of the nodes and time
             * steps.
             */
            std::pair<EvidenceSet, EvidenceSet>
            generate_particles(const EvidenceSet& new_data);

            /**
             * Generates particles for a fixed number of time steps in the
             * future. This procedure returns the samples generated but does not
             * store the last samples in the nodes of the template dbn.
             * Therefore, it generates samples in the future but keeps the
             * particles generated until then for the observations received.
             *
             * @param num_time_steps: number of time step to generate particles
             * in the future
             *
             * @return Samples generated for the next time steps requested.
             */
            EvidenceSet forward_particles(int num_time_steps);

            /**
             * Prepare to start generating particles from scratch.
             */
            void clear_cache();

            //------------------------------------------------------------------
            // Getters & Setters
            //------------------------------------------------------------------

            void set_show_progress(bool show_progress);

          private:
            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            /**
             * Empty constructor to be called by the clone function.
             */
            ParticleFilter();

            /**
             * If the time step is bigger than the number of time steps in the
             * template DBN, we move the particles to the nodes in the before
             * last time step of the template DBN so that in the next time step
             * iteration they become parents of the nodes being sampled.
             *
             * @param time_step: current time step which particles are being
             * generated to
             */
            void move_particles_back_in_time(int time_step);

            /**
             * Creates a short DBN formed by nodes until at most time step 2.
             * After that time step the structure is repeatable.
             *
             */
            void create_template_dbn();

            /**
             * Move particles to the next time step by the underlying process.
             *
             * @param new_data: evidence
             * @param time_step: time step of the particles
             * @param processing_block: block of particles to process
             * @param random_generator: random number generator
             */
            void predict(const EvidenceSet& new_data,
                         int time_step,
                         const ProcessingBlock& processing_block,
                         std::shared_ptr<gsl_rng>& random_generator);

            /**
             * Populate nodes in the evidence set with corresponding values.
             *
             * @param node: node
             * @param data: evidence
             * @param processing_block: rows to process
             */
            void fix_evidence(const RVNodePtr& node,
                              const Eigen::MatrixXd& data,
                              const ProcessingBlock& processing_block);

            /**
             * Resample particles according to observations.
             *
             * @param new_data: evidence
             * @param time_step: time step of the particles
             * @param sampled_particles: indices of the particles to maintain
             *
             * @return Particles generated in the time step
             */
            EvidenceSet resample(const EvidenceSet& new_data,
                                 int time_step,
                                 const Eigen::VectorXi& sampled_particles);

            /**
             * Use observations to weigh most likely particles and sample
             * particles from a discrete distribution given by the normalized
             * wrights.
             *
             * @param time_step: time step of the particles
             * @param new_data: observations
             *
             * @return Indices of particles sampled
             */
            Eigen::VectorXi
            weigh_and_sample_particles(int time_step,
                                       const EvidenceSet& new_data) const;

            /**
             * Shuffles posterior weights accumulated for marginal nodes.
             * Marginal samples do not need to be shuffled because they will be
             * re-sampled from their posterior in the rao-blackwellization
             * phase. We need, however, to shuffle posterior weights because
             * they determine the marginal node's posterior.
             *
             * @param node: marginal node
             * @param sampled_particles: indices of samples to select
             */
            void shuffle_marginal_posterior_weights(
                const RVNodePtr& node,
                const Eigen::VectorXi& sampled_particles);

            /**
             * Shuffles node's samples and the samples from its previous copies
             * as well. Sampling the copy of the node in the previous time step
             * is necessary for correct execution of the rao-blackwellization
             * process. Because transition distributions depend on samples from
             * the previous time step to be correctly addressed.
             *
             * @param node: node
             * @param sampled_particles: indices of samples to select
             */
            void
            shuffle_node_and_previous(const RVNodePtr& node,
                                      const Eigen::VectorXi& sampled_particles);

            /**
             * Shuffles posterior weights of the last left segment for a node
             * that is controlled by a timer.
             *
             * @param node: time controlled node
             * @param sampled_particles: indices of weights to select
             */
            void shuffle_timed_node_left_segment_distributions(
                const RVNodePtr& node,
                const Eigen::VectorXi& sampled_particles);

            /**
             * Updates the particles with samples from marginal nodes'
             * posterior distribution and stores updated posterior weights. This
             * is called Rao-Blackwellization process, commonly used in SLAM for
             * map estimation.
             *
             * @param time_step: time step of the inference process
             * @param particles: resampled particles
             * @param sampled_particles: indices of theparticles to maintain
             *
             * @return Marginal probabilities
             */
            EvidenceSet apply_rao_blackwellization(
                int time_step,
                EvidenceSet& particles,
                const Eigen::VectorXi& sampled_particles);

            /**
             * Gets p (child_timer | parent) in a given time step in log scale.
             *
             * @param parent_node: parent of child_timer
             * @param child_timer: timer node
             *
             * @return log(p (child_timer | parent))
             */
            Eigen::MatrixXd
            get_segment_log_weights(const RVNodePtr& parent_node,
                                    const TimerNodePtr& child_timer) const;

            /**
             * Updates the indices of the distributions at the beginning of the
             * last segment for a timer that is child of a marginal node.
             *
             * @param parent_node: parent of child_timer
             * @param child_timer: timer node
             */
            void update_marginal_left_segment_distributions(
                const RVNodePtr& parent_node, const TimerNodePtr& child_timer);

            /**
             * Shuffle the rows according to a list of row indices. There
             * can be duplicate rows.
             *
             * @param matrix: matrix to be shuffled
             * @param rows: list of row indices
             *
             * @return New matrix formed by the rows of the original matrix
             * in the list of row indices
             */
            Eigen::MatrixXd shuffle_rows(const Eigen::MatrixXd& matrix,
                                         const Eigen::VectorXi& rows) const;

            /**
             * Shuffle the elements of a vector according to a list of indices.
             * There can be duplicate indices.
             *
             * @param original_vector: vector to be shuffled
             * @param indices: list of indices
             *
             * @return New vector formed by the indices of the original vector
             * in the list of indices
             */
            Eigen::VectorXi shuffle_rows(const Eigen::VectorXi& matrix,
                                         const Eigen::VectorXi& indices) const;

            /**
             * Fills a subset of rows in shuffled matrix in a single thread.
             *
             * @param processing_block: subset of rows to be processed by the
             * thread
             * @param original_matrix: original matrix
             * @param shuffled_matrix: matrix of shuffled rows (it will be
             * completely shuffled after all threads finish their job
             * @param shuffled_matrix_mutex: mutex to control writing in the
             * shuffled matrix
             * @param rows: row indices of the original matrix to be placed in
             * the shuffled matrix
             *
             */
            void
            run_shuffle_rows_thread(const std::pair<int, int>& processing_block,
                                    const Eigen::MatrixXd& original_matrix,
                                    Eigen::MatrixXd& shuffled_matrix,
                                    const Eigen::VectorXi& rows,
                                    std::mutex& shuffled_matrix_mutex) const;

            /**
             * Update the index of the distributions at the beginning of the
             * last segment for time controlled nodes.
             *
             * @param time_step: time step of the inference
             */
            void update_left_segment_distribution_indices(int time_step);

            /**
             * Gets the children of a marginal node at a given time step.
             *
             * @param marginal_node: marginal node
             * @param time_step: time step
             *
             * @return
             */
            RVNodePtrVec
            get_marginal_node_children(const RVNodePtr& marginal_node,
                                       int time_step) const;

            //------------------------------------------------------------------
            // Data members
            //------------------------------------------------------------------
            static inline int LAST_TEMPLATE_TIME_STEP = 2;

            multithread::ThreadPool thread_pool;

            ProcessingBlocks processing_blocks;

            DynamicBayesNet original_dbn;

            DynamicBayesNet template_dbn;

            int num_particles;

            std::vector<std::shared_ptr<gsl_rng>> random_generators_per_job;

            // Last time step for which particles were generated and stored in
            // the nodes of the template DBN.
            int last_time_step = -1;

            bool show_progress = false;

            std::unordered_set<std::string> data_node_labels;

            // List of nodes that will be rao-blackwellized (marginalized) in
            // order
            RVNodePtrVec marginal_nodes;

            // Label of the nodes that will be marginalized
            std::unordered_set<std::string> marginal_set;

            // Last marginals computed per marginalizable node
            std::unordered_map<std::string, Tensor3> previous_marginals;

            // Posterior weights updated per particle of nodes being
            // marginalized
            std::unordered_map<std::string, Eigen::MatrixXd>
                cum_marginal_posterior_log_weights;

            // Indices of the duration distribution at the beginning of the last
            // time controlled node's segment
            std::unordered_map<std::string, Eigen::VectorXi>
                last_left_segment_distribution_indices;

            // Indices of the duration distribution at the beginning of the last
            // segment for the first value of a specific marginal node
            std::unordered_map<std::string,
                               std::unordered_map<std::string, Eigen::VectorXi>>
                last_left_segment_marginal_nodes_distribution_indices;

            std::unordered_set<std::string> time_controlled_node_set;
        };

    } // namespace model
} // namespace tomcat
