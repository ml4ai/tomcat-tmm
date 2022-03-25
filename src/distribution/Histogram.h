#pragma once

#include <gsl/gsl_histogram.h>

#include "distribution/Distribution.h"
#include "utils/Definitions.h"

namespace tomcat {
    namespace model {

        /**
         * Histogram distribution with given samples.
         */
        class Histogram : public Distribution {
          public:
            //------------------------------------------------------------------
            // Constructors & Destructor
            //------------------------------------------------------------------

            /**
             * Creates an instance of an histogram distribution for node
             * dependent parameters.
             *
             * @param samples: histogram values sampled from this
             * distribution
             */
            Histogram(const std::shared_ptr<Node>& samples, int num_bins = 10);

            /**
             * Creates an instance of an histogram distribution for node
             * dependent parameters.
             *
             * @param samples: histogram values sampled from this
             * distribution
             */
            Histogram(std::shared_ptr<Node>&& samples, int num_bins = 10);

            /**
             * Creates an instance of an histogram distribution by transforming
             * a numerical vector of samples into a constant node to keep
             * static and node dependent distributions compatible.
             *
             * @param samples: Mean and variance of a histogram
             * distribution
             * @param num_bins: number of bins to use in the construction of the
             * histogram
             */
            Histogram(const Eigen::VectorXd& samples, int num_bins = 10);

            /**
             * Creates an instance of an histogram distribution by transforming
             * a numerical vector of samples into a constant node to keep
             * static and node dependent distributions compatible.
             *
             * @param samples: Mean and variance of a histogram
             * distribution
             * @param num_bins: number of bins to use in the construction of the
             * histogram
             */
            Histogram(Eigen::VectorXd&& samples, int num_bins = 10);

            ~Histogram() noexcept;

            //------------------------------------------------------------------
            // Copy & Move constructors/assignments
            //------------------------------------------------------------------
            Histogram(const Histogram& histogram);

            Histogram& operator=(const Histogram& histogram);

            Histogram(Histogram&&) = default;

            Histogram& operator=(Histogram&&) = default;

            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------
            Eigen::VectorXd
            sample(const std::shared_ptr<gsl_rng>& random_generator,
                   int parameter_idx) const override;

            /**
             * Undefined for an histogram distribution.
             */
            Eigen::VectorXd
            sample(const std::shared_ptr<gsl_rng>& random_generator,
                   const Eigen::VectorXd& weights) const override;

            /**
             * Undefined for an histogram distribution.
             */
            Eigen::VectorXd
            sample(const std::shared_ptr<gsl_rng>& random_generator,
                   const Eigen::VectorXd& weights,
                   double replace_by_weight) const override;

            /**
             * Undefined for an histogram distribution.
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
             * Undefined for an histogram distribution.
             */
            void update_from_posterior(
                const Eigen::VectorXd& posterior_weights) override;

          private:
            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            /**
             * Copies the content of another histogram distribution to this one.
             *
             * @param other: other histogram distribution.
             */
            void copy(const Histogram& other);

            /**
             * Builds a histogram based on values from histogram samples. The
             * bins are uniformly distributed.
             *
             * @param histogram_samples: values to build the histogram
             *
             * @return Histogram
             */
            std::unique_ptr<gsl_histogram>
            build_histogram(const Eigen::VectorXd& histogram_samples) const;

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
             * Returns a list of histogram samples associated to the
             * distribution.
             *
             * @param samples_idx: the index of the samples assignment
             * to use in case the distribution depend on parameter nodes with
             * multiple assignments. If the parameter has single assignment,
             * that is the one being used regardless of the value informed in
             * this argument.
             *
             * @return histogram constructed with the histogram values.
             */
            std::unique_ptr<gsl_histogram> get_histogram(int samples_idx) const;

            /**
             * Generate a sample using the GSL library.
             *
             * @param random_generator: random number generator
             * @param reference_histogram_pdf: histogram pdf structure used for
             * sampling
             *
             * @return A sample from an histogram distribution.
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
