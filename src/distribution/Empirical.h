#pragma once

#include <gsl/gsl_histogram.h>

#include "distribution/Distribution.h"
#include "utils/Definitions.h"

namespace tomcat {
    namespace model {

        /**
         * Empirical distribution with given samples.
         */
        class Empirical : public Distribution {
          public:
            //------------------------------------------------------------------
            // Constructors & Destructor
            //------------------------------------------------------------------

            /**
             * Creates an instance of an empirical distribution for node
             * dependent parameters.
             *
             * @param empirical_samples: empirical values sampled from this
             * distribution
             */
            Empirical(const std::shared_ptr<Node>& empirical_samples,
                      int num_bins = 10);

            /**
             * Creates an instance of an empirical distribution for node
             * dependent parameters.
             *
             * @param empirical_samples: empirical values sampled from this
             * distribution
             */
            Empirical(std::shared_ptr<Node>&& empirical_samples,
                      int num_bins = 10);

            /**
             * Creates an instance of an empirical distribution by transforming
             * a numerical vector of samples into a constant node to keep
             * static and node dependent distributions compatible.
             *
             * @param empirical_samples: Mean and variance of a empirical
             * distribution
             * @param num_bins: number of bins to use in the construction of the
             * histogram
             */
            Empirical(const Eigen::VectorXd& empirical_samples,
                      int num_bins = 10);

            /**
             * Creates an instance of an empirical distribution by transforming
             * a numerical vector of samples into a constant node to keep
             * static and node dependent distributions compatible.
             *
             * @param empirical_samples: Mean and variance of a empirical
             * distribution
             * @param num_bins: number of bins to use in the construction of the
             * histogram
             */
            Empirical(Eigen::VectorXd&& empirical_samples, int num_bins = 10);

            ~Empirical() noexcept;

            //------------------------------------------------------------------
            // Copy & Move constructors/assignments
            //------------------------------------------------------------------
            Empirical(const Empirical& Empirical);

            Empirical& operator=(const Empirical& Empirical);

            Empirical(Empirical&&) = default;

            Empirical& operator=(Empirical&&) = default;

            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------
            Eigen::VectorXd
            sample(const std::shared_ptr<gsl_rng>& random_generator,
                   int parameter_idx) const override;

            /**
             * Undefined for an empirical distribution.
             */
            Eigen::VectorXd
            sample(const std::shared_ptr<gsl_rng>& random_generator,
                   const Eigen::VectorXd& weights) const override;

            /**
             * Undefined for an empirical distribution.
             */
            Eigen::VectorXd
            sample(const std::shared_ptr<gsl_rng>& random_generator,
                   const Eigen::VectorXd& weights,
                   double replace_by_weight) const override;

            /**
             * Undefined for an empirical distribution.
             */
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

            /**
             * Undefined for an empirical distribution.
             */
            void update_from_posterior(
                const Eigen::VectorXd& posterior_weights) override;

          private:
            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            /**
             * Copies the content of another empirical distribution to this one.
             *
             * @param other: other empirical distribution.
             */
            void copy(const Empirical& other);

            /**
             * Builds a histogram based on values from empirical samples. The
             * bins are uniformly distributed.
             *
             * @param empirical_samples: values to build the histogram
             *
             * @return Histogram
             */
            std::unique_ptr<gsl_histogram>
            build_histogram(const Eigen::VectorXd& empirical_samples) const;

            /**
             * Builds a structure to compute pdfs and sample from a reference
             * histogram.
             *
             * @param reference_histogram: reference histogram
             *
             * @return Histogram pdf structure
             */
            std::unique_ptr<gsl_histogram_pdf> build_histogram_pdf(
                const std::unique_ptr<gsl_histogram>& reference_histogram)
                const;

            /**
             * Builds a vector of cdfs from a reference histogram.
             *
             * @param reference_histogram: reference histogram
             *
             * @return Histogram pdf structure
             */
            std::vector<double> build_histogram_cdf(
                const std::unique_ptr<gsl_histogram>& reference_histogram)
                const;

            /**
             * Returns a list of empirical samples associated to the
             * distribution.
             *
             * @param samples_idx: the index of the samples assignment
             * to use in case the distribution depend on parameter nodes with
             * multiple assignments. If the parameter has single assignment,
             * that is the one being used regardless of the value informed in
             * this argument.
             *
             * @return histogram constructed with the empirical values.
             */
            std::unique_ptr<gsl_histogram> get_histogram(int samples_idx) const;

            /**
             * Generate a sample using the GSL library.
             *
             * @param random_generator: random number generator
             * @param reference_histogram_pdf: histogram pdf structure used for
             * sampling
             *
             * @return A sample from an empirical distribution.
             */
            Eigen::VectorXd
            sample_from_gsl(const std::shared_ptr<gsl_rng>& random_generator,
                            const std::unique_ptr<gsl_histogram_pdf>&
                                reference_histogram_pdf) const;

            //------------------------------------------------------------------
            // Data members
            //------------------------------------------------------------------
            int num_bins;
            int num_samples;
            std::unique_ptr<gsl_histogram> histogram;
            std::unique_ptr<gsl_histogram_pdf> histogram_pdf;
            std::vector<double> histogram_cdf;
        };

    } // namespace model
} // namespace tomcat
