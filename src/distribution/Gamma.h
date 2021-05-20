#pragma once

#include "distribution/Distribution.h"
#include "utils/Definitions.h"

namespace tomcat {
    namespace model {

        /**
         * Gamma distribution with parameters a alpha and beta defined by two
         * nodes, which can be constant or random variables.
         */
        class Gamma : public Distribution {
          public:
            //------------------------------------------------------------------
            // Types, Enums & Constants
            //------------------------------------------------------------------
            enum PARAMETER_INDEX { alpha, beta };

            //------------------------------------------------------------------
            // Constructors & Destructor
            //------------------------------------------------------------------

            /**
             * Creates an instance of a Gamma distribution for node
             * dependent parameters.
             *
             * @param parameters: nodes which the assignments define the
             * parameters of the distribution
             */
            Gamma(const std::shared_ptr<Node>& alpha,
                  const std::shared_ptr<Node>& beta);

            /**
             * Creates an instance of a Gamma distribution for node
             * dependent parameters.
             *
             * @param parameters: nodes which the assignments define the
             * parameters of the distribution
             */
            Gamma(std::shared_ptr<Node>&& alpha, std::shared_ptr<Node>&& beta);

            /**
             * Creates an instance of a Gamma distribution by transforming
             * embedding its parameters into a constant node for alpha and
             * another to beta in order to keep static and node  dependent
             * distributions compatible.
             *
             * @param alpha: alpha parameter
             * @param beta: beta parameter
             */
            Gamma(unsigned int alpha, unsigned int beta);

            ~Gamma();

            //------------------------------------------------------------------
            // Copy & Move constructors/assignments
            //------------------------------------------------------------------
            Gamma(const Gamma& gamma);

            Gamma& operator=(const Gamma& gamma);

            Gamma(Gamma&&) = default;

            Gamma& operator=(Gamma&&) = default;

            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------
            Eigen::VectorXd
            sample(const std::shared_ptr<gsl_rng>& random_generator,
                   int parameter_idx) const override;

            /**
             * Generates a sample from a Gamma distribution with scaled
             * parameters.
             *
             * @param random_generator: random number generator
             * @param weights: weights used to scale the parameters
             *
             * @return Sample from a scaled Gamma distribution.
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

            double get_pdf(double value) const override;

            double get_cdf(double value, bool reverse) const override;

            std::unique_ptr<Distribution> clone() const override;

            std::string get_description() const override;

            int get_sample_size() const override;

          private:
            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            /**
             * Returns alpha and beta from assignments of the nodes that
             * represent these parameters.
             *
             * @param parameter_idx: the index of the parameter assignment
             * to use in case the distribution depend on parameter nodes with
             * multiple assignments. If the parameter has single assignment,
             * that is the one being used regardless of the value informed in
             * this argument.
             * @return Vector of containing the alpha and the beta of the
             * distribution.
             */
            Eigen::VectorXd get_parameters(int parameter_idx) const;

            /**
             * Generate a sample using the GSL library.
             *
             * @param random_generator: random number generator
             * @param alpha: alpha or weighted alpha
             * @param beta: beta or weighted beta
             * @return A sample from a Gamma distribution.
             */
            Eigen::VectorXd
            sample_from_gsl(const std::shared_ptr<gsl_rng>& random_generator,
                            double alpha,
                            double beta) const;
        };

    } // namespace model
} // namespace tomcat
