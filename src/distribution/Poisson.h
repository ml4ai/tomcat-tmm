#pragma once

#include "distribution/Distribution.h"
#include "pgm/Node.h"
#include "utils/Definitions.h"

namespace tomcat {
    namespace model {

        /**
         * Class to represent a Poisson distribution.
         */
        class Poisson : public Distribution {
          public:
            //------------------------------------------------------------------
            // Constructors & Destructor
            //------------------------------------------------------------------

            /**
             * Creates an instance of a Poisson distribution for node
             * dependent mean lambda.
             *
             * @param lambda: node which the assignment defines the mean lambda
             */
            Poisson(const std::shared_ptr<Node>& lambda);

            /**
             * Creates an instance of a Poisson distribution for node
             * dependent mean lambda.
             *
             * @param lambda: node which the assignment defines the mean lambda
             */
            Poisson(std::shared_ptr<Node>&& lambda);

            /**
             * Creates an instance of a Poisson distribution by transforming
             * a numerical mean lambda a constant node to keep
             * static and node dependent distributions compatible.
             *
             * @param lambda: mean of the Poisson distribution
             */
            Poisson(double lambda);

            ~Poisson();

            //------------------------------------------------------------------
            // Copy & Move constructors/assignments
            //------------------------------------------------------------------
            Poisson(const Poisson& poisson);

            Poisson& operator=(const Poisson& poisson);

            Poisson(Poisson&&) = default;

            Poisson& operator=(Poisson&&) = default;

            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------
            Eigen::VectorXd
            sample(const std::shared_ptr<gsl_rng>& random_generator,
                   int parameter_idx) const override;

            /**
             * Generates a sample from a Poisson distribution with scaled
             * mean.
             *
             * @param random_generator: random number generator
             * @param weight: weight used to scale the mean
             *
             * @return Sample from a scaled Poisson distribution.
             */
            Eigen::VectorXd
            sample(const std::shared_ptr<gsl_rng>& random_generator,
                   const Eigen::VectorXd& weight) const override;

            Eigen::VectorXd
            sample(const std::shared_ptr<gsl_rng>& random_generator,
                   const Eigen::VectorXd& weights,
                   double replace_by_weight) const override;

            Eigen::VectorXd sample_from_conjugacy(
                const std::shared_ptr<gsl_rng>& random_generator,
                int parameter_idx,
                const Eigen::VectorXd& sufficient_statistics) const override;

            double get_pdf(const Eigen::VectorXd& value) const override;

            double get_pdf(double value) const override;

            double get_cdf(double value, bool reverse) const override;

            std::unique_ptr<Distribution> clone() const override;

            std::string get_description() const override;

            int get_sample_size() const override;

            void
            update_from_posterior(const Eigen::VectorXd& posterior_weights) override {}

          private:
            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            /**
             * Generate a sample using the GSL library.
             *
             * @param random_generator: random number generator
             * @param lambda: mean or scaled mean of the distribution
             *
             * @return A sample from a Poisson distribution.
             */
            Eigen::VectorXd
            sample_from_gsl(const std::shared_ptr<gsl_rng>& random_generator,
                            double lambda) const;
        };

    } // namespace model
} // namespace tomcat
