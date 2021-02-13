#pragma once

#include "distribution/Distribution.h"
#include "pgm/Node.h"
#include "utils/Definitions.h"

namespace tomcat {
    namespace model {

        /**
         * Class to represent a Categorical distribution with discrete
         * probabilities.
         */
        class Categorical : public Distribution {
          public:
            //------------------------------------------------------------------
            // Constructors & Destructor
            //------------------------------------------------------------------

            /**
             * Creates an instance of a categorical distribution for node
             * dependent probabilities.
             *
             * @param probabilities: node which the assignment defines the set
             * of probabilities of the distribution
             */
            Categorical(const std::shared_ptr<Node>& probabilities);

            /**
             * Creates an instance of a categorical distribution for node
             * dependent probabilities.
             *
             * @param probabilities: node which the assignment defines the set
             * of probabilities of the distribution
             */
            Categorical(std::shared_ptr<Node>&& probabilities);

            /**
             * Creates an instance of a categorical distribution by transforming
             * a numerical vector of probabilities into a constant node to keep
             * static and node dependent distributions compatible.
             *
             * @param probabilities: Vector of constant probabilities
             */
            Categorical(const Eigen::VectorXd& probabilities);

            /**
             * Creates an instance of a categorical distribution by transforming
             * a numerical vector of probabilities into a constant node to keep
             * static and node dependent distributions compatible.
             *
             * @param probabilities: Vector of constant probabilities
             */
            Categorical(const Eigen::VectorXd&& probabilities);

            ~Categorical();

            //------------------------------------------------------------------
            // Copy & Move constructors/assignments
            //------------------------------------------------------------------
            Categorical(const Categorical& categorical);

            Categorical& operator=(const Categorical& categorical);

            Categorical(Categorical&&) = default;

            Categorical& operator=(Categorical&&) = default;

            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------
            Eigen::VectorXd
            sample(const std::shared_ptr<gsl_rng>& random_generator,
                   int parameter_idx) const override;

            /**
             * Generates a sample from a categorical distribution with scaled
             * probabilities.
             *
             * @param random_generator: random number generator
             * @param weights: weights used to scale the probabilities
             *
             * @return Sample from a scaled categorical distribution.
             */
            Eigen::VectorXd
            sample(const std::shared_ptr<gsl_rng>& random_generator,
                   const Eigen::VectorXd& weights) const override;

            Eigen::VectorXd
            sample(const std::shared_ptr<gsl_rng>& random_generator,
                   const Eigen::VectorXd& weights,
                   double replace_by_weight) const override;

            Eigen::VectorXd sample_from_conjugacy(
                const std::shared_ptr<gsl_rng>& random_generator,
                int parameter_idx,
                const Eigen::VectorXd& sufficient_statistics) const override;

            double get_pdf(const Eigen::VectorXd& value) const override;

            double get_cdf(double value, bool reverse) const override;

            std::unique_ptr<Distribution> clone() const override;

            std::string get_description() const override;

            int get_sample_size() const override;

          private:
            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            /**
             * Generate a sample using the GSL library.
             *
             * @param random_generator: random number generator
             * @param parameters: probabilities or weighted probabilities
             * @return A sample from a categorical distribution.
             */
            Eigen::VectorXd
            sample_from_gsl(const std::shared_ptr<gsl_rng>& random_generator,
                            const Eigen::VectorXd& parameters) const;

            /**
             * Returns the index of a sampled value from a one-hot-encode array.
             *
             * @param sample_array: one-hot-encode sample
             * @param array_size: size of the one-hot-encode sample
             * @return Index containing 1 in an one-hot-encode array
             */
            unsigned int get_sample_index(const unsigned int* sample_array,
                                          size_t array_size) const;
        };

    } // namespace model
} // namespace tomcat
