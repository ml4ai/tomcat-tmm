#include "Histogram.h"

#include <gsl/gsl_randist.h>

#include "pgm/NumericNode.h"

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        Histogram::Histogram(const shared_ptr<Node>& samples, int num_bins)
            : Distribution({samples}), num_bins(num_bins) {}

        Histogram::Histogram(shared_ptr<Node>&& samples, int num_bins)
            : Distribution({move(samples)}), num_bins(num_bins) {}

        Histogram::Histogram(const Eigen::VectorXd& samples, int num_bins)
            : num_bins(num_bins) {

            this->num_samples = samples.size();
            this->histogram = this->build_histogram(samples);
            this->histogram_pdf = this->build_histogram_pdf(this->histogram);
            this->histogram_cdf = this->build_histogram_cdf(this->histogram);
            NumericNode samples_node(samples.transpose());

            this->parameters.push_back(
                make_shared<NumericNode>(move(samples_node)));
        }

        Histogram::Histogram(Eigen::VectorXd&& samples, int num_bins)
            : num_bins(num_bins) {

            this->num_samples = samples.size();
            this->histogram = this->build_histogram(samples);
            this->histogram_pdf = this->build_histogram_pdf(this->histogram);
            this->histogram_cdf = this->build_histogram_cdf(this->histogram);
            NumericNode samples_node(samples.transpose());

            this->parameters.push_back(
                make_shared<NumericNode>(move(samples_node)));
        }

        Histogram::~Histogram() noexcept {
            if (this->histogram) {
                auto* temp_histogram = this->histogram.get();
                this->histogram.release();
                gsl_histogram_free(temp_histogram);

                auto* temp_histogram_pdf = this->histogram_pdf.get();
                this->histogram_pdf.release();
                gsl_histogram_pdf_free(temp_histogram_pdf);
            }
        }

        //----------------------------------------------------------------------
        // Copy & Move constructors/assignments
        //----------------------------------------------------------------------
        Histogram::Histogram(const Histogram& Histogram) {
            this->copy(Histogram);
        }

        Histogram& Histogram::operator=(const Histogram& Histogram) {
            this->copy(Histogram);
            return *this;
        }

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        void Histogram::copy(const Histogram& other) {
            Distribution::copy(other);
            this->num_bins = other.num_bins;
            this->num_samples = other.num_samples;
            this->histogram_cdf = other.histogram_cdf;
            if (other.histogram) {
                this->histogram = make_unique<gsl_histogram>(
                    *gsl_histogram_clone(other.histogram.get()));
            }
        }

        unique_ptr<gsl_histogram>
        Histogram::build_histogram(const Eigen::VectorXd& samples) const {
            double min_value = samples.minCoeff();
            double max_value = samples.maxCoeff() + 1;

            gsl_histogram* new_histogram = gsl_histogram_alloc(this->num_bins);
            gsl_histogram_set_ranges_uniform(
                new_histogram, min_value, max_value);
            for (int i = 0; i < samples.size(); i++) {
                gsl_histogram_increment(new_histogram, samples(i));
            }

            return make_unique<gsl_histogram>(*new_histogram);
        }

        unique_ptr<gsl_histogram_pdf> Histogram::build_histogram_pdf(
            const unique_ptr<gsl_histogram>& reference_histogram) const {

            gsl_histogram_pdf* new_histogram_pdf =
                gsl_histogram_pdf_alloc(reference_histogram->n);
            gsl_histogram_pdf_init(new_histogram_pdf,
                                   reference_histogram.get());

            return make_unique<gsl_histogram_pdf>(*new_histogram_pdf);
        }

        vector<double> Histogram::build_histogram_cdf(
            const unique_ptr<gsl_histogram>& reference_histogram) const {

            vector<double> cdfs(this->num_bins);

            for (int i = 0; i < this->num_bins; i++) {
                cdfs[i] = gsl_histogram_get(reference_histogram.get(), i);

                if (i > 0) {
                    cdfs[i] += cdfs[i - 1];
                }
            }

            for (int i = 0; i < this->num_bins; i++) {
                cdfs[i] /= cdfs[this->num_bins - 1];
            }

            return cdfs;
        }

        Eigen::VectorXd
        Histogram::sample(const shared_ptr<gsl_rng>& random_generator,
                          int parameter_idx) const {

            Eigen::VectorXd sample;
            if (this->histogram) {
                // The Histogram values are fixed and the histogram was already
                // precomputed. We use it.

                sample = this->sample_from_gsl(random_generator,
                                               this->histogram_pdf);
            }
            else {
                // Histograms are created on-the-fly based on the must
                // up-to-date Histogram samples in the parameter node.
                unique_ptr<gsl_histogram> temp_histogram =
                    this->build_histogram(
                        this->parameters[0]->get_assignment().row(
                            parameter_idx));
                unique_ptr<gsl_histogram_pdf> temp_histogram_pdf =
                    this->build_histogram_pdf(temp_histogram);
                sample =
                    this->sample_from_gsl(random_generator, temp_histogram_pdf);
                gsl_histogram_pdf_free(temp_histogram_pdf.get());
                gsl_histogram_free(temp_histogram.get());
            }

            return sample;
        }

        Eigen::VectorXd
        Histogram::sample_from_gsl(const shared_ptr<gsl_rng>& random_generator,
                                   const unique_ptr<gsl_histogram_pdf>&
                                       reference_histogram_pdf) const {

            double random_number = gsl_rng_uniform(random_generator.get());
            double sample = gsl_histogram_pdf_sample(
                reference_histogram_pdf.get(), random_number);

            Eigen::VectorXd sample_vector(1);
            sample_vector(0) = sample;

            return sample_vector;
        }

        Eigen::VectorXd
        Histogram::sample(const shared_ptr<gsl_rng>& random_generator,
                          const Eigen::VectorXd& weights) const {

            throw TomcatModelException(
                "Not implemented for Histogram distributions.");
        }

        Eigen::VectorXd
        Histogram::sample(const std::shared_ptr<gsl_rng>& random_generator,
                          const Eigen::VectorXd& weights,
                          double replace_by_weight) const {

            throw TomcatModelException(
                "Not implemented for Histogram distributions.");
        }

        Eigen::VectorXd Histogram::sample_from_conjugacy(
            const shared_ptr<gsl_rng>& random_generator,
            int parameter_idx,
            const Eigen::VectorXd& sufficient_statistics) const {

            throw TomcatModelException(
                "Not implemented for Histogram distributions.");
        }

        double Histogram::get_pdf(const Eigen::VectorXd& value) const {
            return this->get_pdf(value(0));
        }

        double Histogram::get_pdf(double value) const {
            double pdf;

            if (this->histogram) {
                size_t* bin = new size_t(0);
                if (gsl_histogram_find(this->histogram.get(), value, bin) ==
                    GSL_SUCCESS) {
                    pdf = gsl_histogram_get(this->histogram.get(), *bin);
                    double* lb = new double(0);
                    double* ub = new double(0);
                    gsl_histogram_get_range(
                        this->histogram.get(), *bin, lb, ub);

                    double bin_width = *ub - *lb;
                    pdf /= this->num_samples * bin_width;
                }
                else {
                    pdf = 0;
                }
            }
            else {
                // Builds histogram on-the-fly
                Eigen::VectorXd samples =
                    this->parameters[0]->get_assignment().row(0);
                unique_ptr<gsl_histogram> temp_histogram =
                    this->build_histogram(samples);
                size_t* bin = new size_t(0);
                if (gsl_histogram_find(temp_histogram.get(), value, bin) ==
                    GSL_SUCCESS) {
                    pdf = gsl_histogram_get(temp_histogram.get(), *bin) /
                          (double)samples.size();
                    gsl_histogram_free(temp_histogram.get());
                }
                else {
                    pdf = 0;
                }
            }

            return pdf;
        }

        double Histogram::get_cdf(double value, bool reverse) const {
            double cdf;

            if (this->histogram) {
                size_t* bin = new size_t(0);
                if (gsl_histogram_find(this->histogram.get(), value, bin) ==
                    GSL_SUCCESS) {
                    cdf = this->histogram_cdf[*bin];
                }
                else {
                    cdf = 0;
                }
            }
            else {
                // Builds histogram on-the-fly
                unique_ptr<gsl_histogram> temp_histogram =
                    this->build_histogram(
                        this->parameters[0]->get_assignment().row(0));
                auto cdfs = this->build_histogram_cdf(temp_histogram);
                size_t* bin = new size_t(0);
                if (gsl_histogram_find(temp_histogram.get(), value, bin) ==
                    GSL_SUCCESS) {
                    cdf = cdfs[*bin];
                    gsl_histogram_free(temp_histogram.get());
                }
                else {
                    cdf = 0;
                }
            }

            if (reverse) {
                cdf = 1 - cdf;
            }

            return cdf;
        }

        unique_ptr<Distribution> Histogram::clone() const {
            unique_ptr<Histogram> new_distribution =
                make_unique<Histogram>(*this);

            for (auto& parameter : new_distribution->parameters) {
                // Do not clone numeric nodes to allow them to be sharable and
                // improve memory efficiency. Numeric nodes are constant, so
                // sharing them causes no harm to the computation.
                if (parameter->is_random_variable()) {
                    parameter = parameter->clone();
                }
            }

            return new_distribution;
        }

        string Histogram::get_description() const {
            stringstream ss;
            ss << "Histogram(bins = " << this->num_bins << ")";
            return ss.str();
        }

        int Histogram::get_sample_size() const { return 1; }

        void Histogram::update_from_posterior(
            const Eigen::VectorXd& posterior_weights) {

            throw TomcatModelException(
                "Not implemented for Histogram distributions.");
        }

    } // namespace model
} // namespace tomcat
