#pragma once

#include "distribution/Gamma.h"
#include "utils/Definitions.h"

namespace tomcat {
    namespace model {

        /**
         * InverseGamma distribution with parameters a alpha and beta defined by
         * two nodes, which can be constant or random variables.
         */
        class InverseGamma : public Gamma {
          public:
            //------------------------------------------------------------------
            // Constructors & Destructor
            //------------------------------------------------------------------

            /**
             * Creates an instance of a InverseGamma distribution for node
             * dependent parameters.
             *
             * @param alpha: node containing the value of the parameter alpha
             * @param beta: node containing the value of the parameter beta
             */
            InverseGamma(const std::shared_ptr<Node>& alpha,
                         const std::shared_ptr<Node>& beta);

            /**
             * Creates an instance of a InverseGamma distribution for node
             * dependent parameters.
             *
             * @param alpha: node containing the value of the parameter alpha
             * @param beta: node containing the value of the parameter beta
             */
            InverseGamma(std::shared_ptr<Node>&& alpha,
                         std::shared_ptr<Node>&& beta);

            /**
             * Creates an instance of a InverseGamma distribution for node
             * dependent parameters.
             *
             * @param parameters: nodes containing the two parameters (alpha and
             * beta) of a InverseGamma distribution
             */
            InverseGamma(const std::vector<std::shared_ptr<Node>>& parameters);

            /**
             * Creates an instance of a InverseGamma distribution for node
             * dependent parameters.
             *
             * @param parameters: nodes containing the two parameters (alpha and
             * beta) of a InverseGamma distribution
             */
            InverseGamma(std::vector<std::shared_ptr<Node>>&& parameters);

            /**
             * Creates an instance of a InverseGamma distribution by
             * transforming embedding its parameters into a constant node for
             * alpha and another to beta in order to keep static and node
             * dependent distributions compatible.
             *
             * @param alpha: alpha parameter
             * @param beta: beta parameter
             */
            InverseGamma(unsigned int alpha, unsigned int beta);

            ~InverseGamma();

            //------------------------------------------------------------------
            // Copy & Move constructors/assignments
            //------------------------------------------------------------------
            InverseGamma(const InverseGamma& InverseGamma);

            InverseGamma& operator=(const InverseGamma& InverseGamma);

            InverseGamma(InverseGamma&&) = default;

            InverseGamma& operator=(InverseGamma&&) = default;

            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------
            Eigen::VectorXd sample_from_conjugacy(
                const std::shared_ptr<gsl_rng>& random_generator,
                int parameter_idx,
                const Eigen::VectorXd& sufficient_statistics) const override;

            double get_pdf(double value) const override;

            double get_cdf(double value, bool reverse) const override;

            std::unique_ptr<Distribution> clone() const override;

            std::string get_description() const override;

          protected:
            Eigen::VectorXd
            sample_from_gsl(const std::shared_ptr<gsl_rng>& random_generator,
                            double alpha,
                            double beta) const override;
        };

    } // namespace model
} // namespace tomcat
