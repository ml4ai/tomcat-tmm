#pragma once

#include "sampling/Sampler.h"

#include "utils/Definitions.h"

namespace tomcat {
    namespace model {

        class Sampler;

        /**
         * Generates samples by ancestral sampling process. When data is
         * provided for specific nodes, samples are not generated for these
         * nodes and the samples generated for the latent nodes will be
         * conditioned on their parent node's data if provided.
         */
        class AncestralSampler : public Sampler {
          public:
            //------------------------------------------------------------------
            // Constructors & Destructor
            //------------------------------------------------------------------

            /**
             * Creates an instance of the sampler for a given model and random
             * number generator.
             *
             * @param model: DBN
             * @param num_jobs: number of threads created for parallel
             * sampling. If 1, no parallel processing is performed and  the code
             * runs in the main thread.
             */
            AncestralSampler(const std::shared_ptr<DynamicBayesNet>& model,
                             int num_jobs = 1);

            ~AncestralSampler();

            //------------------------------------------------------------------
            // Copy & Move constructors/assignments
            //------------------------------------------------------------------
            AncestralSampler(const AncestralSampler& sampler);

            AncestralSampler& operator=(const AncestralSampler& sampler);

            AncestralSampler(AncestralSampler&&) = default;

            AncestralSampler& operator=(AncestralSampler&&) = default;

            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------
            void sample_latent(const std::shared_ptr<gsl_rng>& random_generator,
                               int num_samples) override;

            void get_info(nlohmann::json& json) const override;

            std::unique_ptr<Sampler> clone() const override;

            std::unordered_set<std::string>
            get_sampled_node_labels() const override;

            //------------------------------------------------------------------
            // Getters & Setters
            //------------------------------------------------------------------
            void set_min_initialization_time_step(int time_step) override;

            void set_equal_samples_time_step_limit(
                int equal_samples_time_step_limit);

          private:
            //------------------------------------------------------------------
            // Structs
            //------------------------------------------------------------------

            // Store the nodes that were sampled by the sampler.
            struct NodeSet {
                NodePtrVec nodes_to_sample;
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

            //------------------------------------------------------------------
            // Data members
            //------------------------------------------------------------------

            // If set, samples up to this time will be the same for all the
            // samples generated. Starting from the next time step, samples will
            // be able to diverge according to a regular ancestral sampling
            // process.
            int equal_samples_time_step_limit = -1;
        };
    } // namespace model
} // namespace tomcat
