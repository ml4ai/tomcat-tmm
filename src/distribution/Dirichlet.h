#pragma once

#include "distribution/Distribution.h"
#include "utils/Definitions.h"

namespace tomcat {
    namespace model {

        /**
         * Dirichlet distribution with parameters defined by the vector
         * \f$\alpha\f$. Each one of the coefficients of this vector is stored
         * in the assignment of a node, which can be constant or a random
         * variable.
         */
        class Dirichlet : public Distribution {

          public:
            //------------------------------------------------------------------
            // Constructors & Destructor
            //------------------------------------------------------------------

            /**
             * Creates an instance of a Dirichlet distribution for node
             * dependent parameters.
             *
             * @param parameters: nodes which the assignments define the set
             * of parameters of the distribution
             */
            Dirichlet(const std::vector<std::shared_ptr<Node>>& alpha);

            /**
             * Creates an instance of a Dirichlet distribution for node
             * dependent parameters.
             *
             * @param parameters: nodes which the assignments define the set
             * of parameters of the distribution
             */
            Dirichlet(std::vector<std::shared_ptr<Node>>&& alpha);

            /**
             * Creates an instance of a dirichlet distribution by transforming
             * a constant parameter vector \f$\alpha\f$ into a list of constant
             * nodes to keep static and node dependent distributions compatible.
             *
             * @param alpha: Parameters of a Dirichlet distribution
             */
            Dirichlet(const Eigen::VectorXd& alpha);

            ~Dirichlet();

            //------------------------------------------------------------------
            // Copy & Move constructors/assignments
            //------------------------------------------------------------------
            Dirichlet(const Dirichlet& dirichlet);

            Dirichlet& operator=(const Dirichlet& dirichlet);

            Dirichlet(Dirichlet&&) = default;

            Dirichlet& operator=(Dirichlet&&) = default;

            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------
            Eigen::VectorXd
            sample(const std::shared_ptr<gsl_rng>& random_generator,
                   int parameter_idx) const override;

            /**
             * Generates a sample from a Dirichlet distribution with scaled
             * parameters.
             *
             * @param random_generator: random number generator
             * @param weights: weights used to scale the parameters
             *
             * @return Sample from a scaled Dirichlet distribution.
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

            void
            update_from_posterior(const Eigen::VectorXd& posterior_weights) override;

          private:
            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            /**
             * Initializes and a constant vector alpha if all parameter nodes
             * are constant to avoid looping over them every time alpha is
             * requested.
             */
            void init_constant_alpha();

            /**
             * Returns the parameter vectors \f$\alpha\f$s formed by the
             * assignments of the nodes in the list of parameters.
             *
             * @param parameter_idx: the index of the parameter assignment
             * to use in case the distribution depend on parameter nodes with
             * multiple assignments. If the parameter has single assignment,
             * that is the one being used regardless of the value informed in
             * this argument.             *
             * @return Vector of containing the coefficients of \f$\alpha\f$
             */
            Eigen::VectorXd get_alpha(int parameter_idx) const;

            /**
             * Generate a sample using the GSL library.
             *
             * @param random_generator: random number generator
             * @param parameters: parameters or weighted parameters
             * @return A sample from a Dirichlet distribution.
             */
            Eigen::VectorXd
            sample_from_gsl(const std::shared_ptr<gsl_rng>& random_generator,
                            const Eigen::VectorXd& parameters) const;

            //------------------------------------------------------------------
            // Data members
            //------------------------------------------------------------------
            Eigen::MatrixXd constant_alpha;
        };

    } // namespace model
} // namespace tomcat
