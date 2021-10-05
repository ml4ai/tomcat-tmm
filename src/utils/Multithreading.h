#pragma once

#include <gsl/gsl_rng.h>
#include <memory>
#include <utility>
#include <vector>

namespace tomcat {
    namespace model {

        typedef std::pair<int, int> ProcessingBlock;
        typedef std::vector<ProcessingBlock> ProcessingBlocks;
        /**
         * Gets initial row and the number of rows to process per
         * thread.
         *
         * @param num_jobs: number of threads
         * @param data_size: number of rows of a matrix to be processed in
         * parallel
         *
         * @return list of row indices and size of sub-matrices to be
         * processed in parallel
         */
        inline ProcessingBlocks
        get_parallel_processing_blocks(int num_jobs, int data_size) {
            std::vector<std::pair<int, int>> processing_blocks;

            int rows_per_job = data_size / num_jobs;
            int i = 0;
            for (; i < rows_per_job * (num_jobs - 1); i += rows_per_job) {
                std::pair<int, int> block = std::make_pair(i, rows_per_job);
                processing_blocks.push_back(move(block));
            }

            // Last processing block
            std::pair<int, int> block = std::make_pair(i, data_size - i);
            processing_blocks.push_back(move(block));

            return processing_blocks;
        }

        /**
         * Creates a new random number generator per thread.
         *
         * @param random_generator: original random number generator
         * @param num_jobs: number of threads
         *
         * @return List of random number generator per job
         */
        inline std::vector<std::shared_ptr<gsl_rng>>
        split_random_generator(const std::shared_ptr<gsl_rng>& random_generator,
                               int num_jobs) {
            std::vector<std::shared_ptr<gsl_rng>> random_generators_per_job;
            random_generators_per_job.reserve(num_jobs);

            if (num_jobs == 1) {
                // Use the main thread random generator
                random_generators_per_job.push_back(random_generator);
            }
            else {
                // Create copies of the original random generator
                for (int job = 0; job < num_jobs; job++) {
                    std::shared_ptr<gsl_rng> new_random_gen(
                        gsl_rng_alloc(random_generator->type));
                    // New seed
                    long seed = 0;
                    while(seed == 0) {
                        seed = gsl_rng_get(random_generator.get());
                    }

                    gsl_rng_set(new_random_gen.get(), seed);
                    random_generators_per_job.push_back(move(new_random_gen));
                }
            }

            return random_generators_per_job;
        }
    } // namespace model
} // namespace tomcat