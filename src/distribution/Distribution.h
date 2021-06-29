#pragma once

#include <mutex>
#include <utility>

#include <eigen3/Eigen/Dense>
#include <gsl/gsl_rng.h>

#include "pgm/Node.h"
#include "utils/Definitions.h"

namespace tomcat {
    namespace model {

        /**
         * Abstract probability distribution.
         */
        class Distribution {
          public:
            //------------------------------------------------------------------
            // Constructors & Destructor
            //------------------------------------------------------------------

            /**
             * Creates an abstract representation of a distribution.
             */
            Distribution();

            /**
             * Creates an abstract representation of a  distribution for node
             * dependent parameters.
             *
             * @param parameters: nodes which the assignments define the set
             * of parameters of the distribution
             */
            Distribution(const std::vector<std::shared_ptr<Node>>& parameters);

            /**
             * Creates an abstract representation of a distribution for node
             * dependent parameters.
             *
             * @param parameters: nodes which the assignments define the set
             * of parameters of the distribution
             */
            Distribution(std::vector<std::shared_ptr<Node>>&& parameters);

            virtual ~Distribution();

            //------------------------------------------------------------------
            // Copy & Move constructors/assignments
            //------------------------------------------------------------------

            // Copy constructor and assignment should be deleted to avoid
            // implicit slicing and loss of polymorphic behaviour in the
            // subclasses. To deep copy, the clone method must be used.
            Distribution(const Distribution&) = delete;

            Distribution& operator=(const Distribution&) = delete;

            Distribution(Distribution&&) = default;

            Distribution& operator=(Distribution&&) = default;

            //------------------------------------------------------------------
            // Operator overload
            //------------------------------------------------------------------
            friend std::ostream& operator<<(std::ostream& os,
                                            const Distribution& distribution);

            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            /**
             * Replaces parameter nodes in the distribution by the correct
             * copy of the node in an unrolled DBN.
             *
             * @param parameter_nodes_map: mapping between a parameter node's
             * timed name and its concrete object reference in an unrolled DBN
             */
            void update_dependencies(const Node::NodeMap& parameter_nodes_map);

            /**
             * Update the sufficient statistics in the parameter nodes given the
             * collection of values informed.
             *
             * @param values: Values from the data node that depends on
             * the parameter being updated
             */
            void
            update_sufficient_statistics(const std::vector<double>& values);

            /**
             * Returns assignments of the node(s) the distribution depends on. A
             * node can have multiple assignments. Only the first one is
             * returned by this function.
             *
             * @param parameter_idx: the index of the parameter assignment
             * to use in case the distribution depend on parameter nodes with
             * multiple assignments. If the parameter has single assignment,
             * that is the one being used regardless of the value informed in
             * this argument.
             *
             * @return Concrete node assignments.
             */
            Eigen::VectorXd get_values(int parameter_idx) const;

            /**
             * Prints a short description of the distribution.
             *
             * @param os: output stream
             */
            void print(std::ostream& os) const;

            /**
             *
             * @param random_generators_per_job: random number generator to be
             * used by individual threads
             * @param num_samples: number of samples to generate
             * @param parameter_idx: row of the node's assignment that holds the
             * parameters of the distribution
             *
             * @return Matrix of samples generated for the distribution (one
             * sample per row)
             */
            Eigen::MatrixXd sample_many(
                std::vector<std::shared_ptr<gsl_rng>> random_generator_per_job,
                int num_samples,
                int parameter_idx) const;

            //------------------------------------------------------------------
            // Pure virtual functions
            //------------------------------------------------------------------

            /**
             * Draws a sample from the distribution.
             *
             * @param random_generator: random number random_generator
             * @param parameter_idx: the index of the parameter assignment
             * to use in case the distribution depend on parameter nodes with
             * multiple assignments. If the parameter has single assignment,
             * that is the one being used regardless of the value informed in
             * this argument.
             *
             * @return A sample from the distribution.
             */
            virtual Eigen::VectorXd
            sample(const std::shared_ptr<gsl_rng>& random_generator,
                   int parameter_idx = 0) const = 0;

            /**
             * Generates a weighted sample from the distribution.
             *
             * @param random_generator: random number generator
             * @param weights: weights
             *
             * @return Weighted sample.
             */
            virtual Eigen::VectorXd
            sample(const std::shared_ptr<gsl_rng>& random_generator,
                   const Eigen::VectorXd& weights) const = 0;

            /**
             * Generates a weighted sample from the distribution.
             *
             * @param random_generator: random number generator
             * @param weights: weights
             * @param replace_by_weight: the probability of the value passed
             * here is only defined by the weight given to it.
             *
             * @return Weighted sample.
             */
            virtual Eigen::VectorXd
            sample(const std::shared_ptr<gsl_rng>& random_generator,
                   const Eigen::VectorXd& weights,
                   double replace_by_weight) const = 0;

            /**
             * Draws a sample from a posterior computed by conjugacy using
             * sufficient statistics.
             *
             * @param random_generator: random number random_generator
             * @param parameter_idx: the index of the parameter assignment
             * to use in case the distribution depend on parameter nodes with
             * multiple assignments. If the parameter has single assignment,
             * that is the one being used regardless of the value informed in
             * this argument.
             * @param sufficient_statistics: sufficient statistics needed to
             * come up with a posterior for the distribution
             *
             * @return A sample from the posterior distribution
             */
            virtual Eigen::VectorXd sample_from_conjugacy(
                const std::shared_ptr<gsl_rng>& random_generator,
                int parameter_idx,
                const Eigen::VectorXd& sufficient_statistics) const = 0;

            /**
             * Returns the PDF/PMF for a given value.
             *
             * @param value: value
             *
             * @return PDF/PMF
             */
            virtual double get_pdf(const Eigen::VectorXd& value) const = 0;

            /**
             * Returns the PDF/PMF for a given value.
             *
             * @param value: value
             *
             * @return PDF/PMF
             */
            virtual double get_pdf(double value) const = 0;

            /**
             * Returns the CDF for a given value.
             *
             * @param value: value
             * @param reverse: computes p(x > value) instead of p(x <= value)
             *
             * @return PDF/PMF
             */
            virtual double get_cdf(double value, bool reverse) const = 0;

            /**
             * Creates a new unique pointer from a concrete instance of a
             * distribution.
             *
             * @return Pointer to the new distribution.
             */
            virtual std::unique_ptr<Distribution> clone() const = 0;

            /**
             * Returns the size of a sample generated by the distribution.
             *
             * @return Sample size.
             */
            virtual int get_sample_size() const = 0;

            /**
             * Updates a distribution to it's conjugate posterior
             *
             * @param weights: posterior weights if applicable
             */
            virtual void
            update_from_posterior(const Eigen::VectorXd& posterior_weights) = 0;

          protected:
            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            /**
             * Copy data members from another distribution.
             *
             * @param distribution: distribution to copy from
             */
            void copy(const Distribution& distribution);

            //------------------------------------------------------------------
            // Pure virtual functions
            //------------------------------------------------------------------

            /**
             * Returns a short description of the distribution.
             *
             * @return Distribution's description.
             */
            virtual std::string get_description() const = 0;

            //------------------------------------------------------------------
            // Data members
            //------------------------------------------------------------------
            // The assignment of a node defines one of the parameters of the
            // distribution.
            std::vector<std::shared_ptr<Node>> parameters;

          private:
            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            /**
             * Generates samples for a specific processing block in a separate
             * thread.
             *
             * @param random_generators_per_job: random number generator per
             * thread
             * @param parameter_idx: row of the node's assignment that holds the
             * parameters of the distribution
             * @param full_samples: matrix where the samples must be stored
             * @param processing_block: range of the total number of samples
             * that must be generated
             * @param samples_mutex: mutex to lock the matrix of full samples
             * when samples generated by this functions are written to it
             */
            void
            run_sample_thread(std::shared_ptr<gsl_rng> random_generator_per_job,
                              int parameter_idx,
                              Eigen::MatrixXd& full_samples,
                              const std::pair<int, int>& processing_block,
                              std::mutex& samples_mutex) const;
        };

    } // namespace model
} // namespace tomcat
