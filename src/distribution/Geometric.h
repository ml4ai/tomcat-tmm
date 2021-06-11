#pragma once

#include "distribution/Distribution.h"
#include "pgm/Node.h"
#include "utils/Definitions.h"

namespace tomcat {
    namespace model {

        /**
         * Class to represent a Geometric distribution.
         */
        class Geometric : public Distribution {
          public:
            //------------------------------------------------------------------
            // Constructors & Destructor
            //------------------------------------------------------------------

            /**
             * Creates an instance of a Geometric distribution for node
             * dependent probability of success p.
             *
             * @param p: node which the assignment defines the probability p
             */
            Geometric(const std::shared_ptr<Node>& p);

            /**
             * Creates an instance of a Geometric distribution for node
             * dependent probability p.
             *
             * @param p: node which the assignment defines the probability p
             */
            Geometric(std::shared_ptr<Node>&& p);

            /**
             * Creates an instance of a Geometric distribution by transforming
             * a numerical probability p a constant node to keep
             * static and node dependent distributions compatible.
             *
             * @param p: probability of success
             */
            Geometric(const double p);

            ~Geometric();

            //------------------------------------------------------------------
            // Copy & Move constructors/assignments
            //------------------------------------------------------------------
            Geometric(const Geometric& geometric);

            Geometric& operator=(const Geometric& geometric);

            Geometric(Geometric&&) = default;

            Geometric& operator=(Geometric&&) = default;

            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------
            Eigen::VectorXd
            sample(const std::shared_ptr<gsl_rng>& random_generator,
                   int parameter_idx) const override;

            /**
             * Generates a sample from a Geometric distribution with scaled
             * probability.
             *
             * @param random_generator: random number generator
             * @param weight: weight used to scale the probability
             *
             * @return Sample from a scaled Geometric distribution.
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
             * @param p: mean or scaled mean of the distribution
             *
             * @return A sample from a Geometric distribution.
             */
            Eigen::VectorXd
            sample_from_gsl(const std::shared_ptr<gsl_rng>& random_generator,
                            double p) const;
        };

    } // namespace model
} // namespace tomcat
