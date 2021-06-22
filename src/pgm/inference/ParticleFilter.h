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

            ParticleFilter(const ParticleFilter&) = default;

            ParticleFilter& operator=(const ParticleFilter&) = default;

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
             * Update the particles with samples from single time nodes'
             * posterior distribution. Also, update that distribution to be used
             * as a prior in the next time step iteration. This is called
             * Rao-Blackwellization process, commonly used in SLAM for map
             * estimation.
             *
             * @param particles: particles for non-single time nodes
             * @param time_step: time step of the inference process
             *
             * @return Marginal probabilities
             */
            EvidenceSet apply_rao_blackwellization(int time_step);

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
             * Creates a short DBN formed by nodes until at most time step 2.
             * After that time step the structure is repeatable.
             *
             */
            void create_template_dbn();

            /**
             * Move particles to the next time step by the underlying process.
             *
             * @param time_step: time step of the particles
             */
            void elapse(const EvidenceSet& new_data, int time_step);

            /**
             * Resample particles according to observations.
             *
             * @param time_step: time step of the particles
             *
             * @return Particles generated in the time step
             */
            EvidenceSet resample(const EvidenceSet& new_data, int time_step);

            /**
             * Update the forward assignment of a timer node which accounts for
             * the length of a current segment from the left to the right.
             * .
             * @param timer: timer node that is being sampled
             */
            void update_timer_forward_assignment(const TimerNodePtr& timer);

            /**
             * Shuffle the rows according to a list of row indices. There can be
             * duplicate rows.
             *
             * @param matrix: matrix to be shuffled
             * @param rows: list of row indices
             *
             * @return New matrix formed by the rows of the original matrix in
             * the list of row indices
             */
            Eigen::MatrixXd shuffle_rows(const Eigen::MatrixXd& matrix,
                                         const Eigen::VectorXi& rows) const;

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

            //------------------------------------------------------------------
            // Data members
            //------------------------------------------------------------------
            static inline int LAST_TEMPLATE_TIME_STEP = 2;

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

            // Posterior weights updated per particle of nodes being
            // marginalized
            std::unordered_map<std::string, Eigen::MatrixXd>
                marginal_posterior_weights;

            // Indices of the duration distribution at the beginning of the last
            // time controlled node's segment
            std::unordered_map<std::string, Eigen::VectorXi>
                last_left_segment_distribution_indices;

            // Indices of the duration distribution at the beginning of the last
            // segment for the first value of a specific marginal node
            std::unordered_map<std::string,
                               std::unordered_map<std::string, Eigen::VectorXi>>
                last_left_segment_marginal_nodes_distribution_indices;

            std::unordered_set<std::string>  time_controlled_node_set;
        };

    } // namespace model
} // namespace tomcat
