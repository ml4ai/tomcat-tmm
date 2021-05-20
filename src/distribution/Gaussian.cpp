#include "Gaussian.h"

#include <gsl/gsl_randist.h>
#include <gsl/gsl_cdf.h>

#include "pgm/ConstantNode.h"

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        Gaussian::Gaussian(const shared_ptr<Node>& mean,
                           const shared_ptr<Node>& variance)
            : Distribution({mean, variance}) {}

        Gaussian::Gaussian(shared_ptr<Node>&& mean, shared_ptr<Node>&& variance)
            : Distribution({move(mean), move(variance)}) {}

        Gaussian::Gaussian(double mean, double variance) {
            ConstantNode mean_node(mean);
            ConstantNode variance_node(variance);

            this->parameters.push_back(
                make_shared<ConstantNode>(move(mean_node)));
            this->parameters.push_back(
                make_shared<ConstantNode>(move(variance_node)));
        }

        Gaussian::~Gaussian() {}

        //----------------------------------------------------------------------
        // Copy & Move constructors/assignments
        //----------------------------------------------------------------------
        Gaussian::Gaussian(const Gaussian& gaussian) {
            this->copy(gaussian);
        }

        Gaussian& Gaussian::operator=(const Gaussian& gaussian) {
            this->copy(gaussian);
            return *this;
        }

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        Eigen::VectorXd
        Gaussian::sample(const shared_ptr<gsl_rng>& random_generator,
                         int parameter_idx) const {

            Eigen::VectorXd parameters = this->get_parameters(parameter_idx);
            double mean = parameters(PARAMETER_INDEX::mean);
            double variance = parameters(PARAMETER_INDEX::variance);

            return this->sample_from_gsl(random_generator, mean, variance);
        }

        Eigen::VectorXd Gaussian::get_parameters(int parameter_idx) const {
            Eigen::VectorXd means =
                this->parameters[PARAMETER_INDEX::mean]->get_assignment().col(
                    0);
            Eigen::VectorXd variances =
                this->parameters[PARAMETER_INDEX::variance]
                    ->get_assignment()
                    .col(0);

            int mean_idx = means.size() == 1 ? 0 : parameter_idx;
            double mean = means(mean_idx);

            int variance_idx = variances.size() == 1 ? 0 : parameter_idx;
            double variance = variances(variance_idx);

            Eigen::VectorXd parameters(2);
            parameters(PARAMETER_INDEX::mean) = mean;
            parameters(PARAMETER_INDEX::variance) = variance;

            return parameters;
        }

        Eigen::VectorXd
        Gaussian::sample_from_gsl(const shared_ptr<gsl_rng>& random_generator,
                                  double mean,
                                  double variance) const {

            double sample =
                mean + gsl_ran_gaussian(random_generator.get(), sqrt(variance));

            Eigen::VectorXd sample_vector(1);
            sample_vector(0) = sample;

            return sample_vector;
        }

        Eigen::VectorXd
        Gaussian::sample(const shared_ptr<gsl_rng>& random_generator,
                         const Eigen::VectorXd& weights) const {

            Eigen::VectorXd parameters = this->get_parameters(0) * weights;
            double mean = parameters(PARAMETER_INDEX::mean);
            double variance = parameters(PARAMETER_INDEX::variance);

            return this->sample_from_gsl(random_generator, mean, variance);
        }

        Eigen::VectorXd
        Gaussian::sample(const std::shared_ptr<gsl_rng>& random_generator,
                          const Eigen::VectorXd& weights,
                          double replace_by_weight) const {
            throw TomcatModelException("Not defined for continuous "
                                       "distributions.");
        }

        Eigen::VectorXd Gaussian::sample_from_conjugacy(
            const shared_ptr<gsl_rng>& random_generator,
            int parameter_idx,
            const Eigen::VectorXd& sufficient_statistics) const {
            throw invalid_argument("Not implemented yet.");
        }

        double Gaussian::get_pdf(const Eigen::VectorXd& value) const {
            return this->get_pdf(value(0));
        }

        double Gaussian::get_pdf(double value) const {
            Eigen::VectorXd parameters = this->get_parameters(0);
            double mean = parameters(PARAMETER_INDEX::mean);
            double variance = parameters(PARAMETER_INDEX::variance);

            return gsl_ran_gaussian_pdf(value - mean, sqrt(variance));
        }

        double Gaussian::get_cdf(double value, bool reverse) const {
            Eigen::VectorXd parameters = this->get_parameters(0);
            double mean = parameters(PARAMETER_INDEX::mean);
            double variance = parameters(PARAMETER_INDEX::variance);

            double cdf = gsl_cdf_gaussian_P(value - mean, sqrt(variance));
            if (reverse) {
                cdf = 1 - cdf;
            }

            return cdf;
        }

        unique_ptr<Distribution> Gaussian::clone() const {
            unique_ptr<Gaussian> new_distribution =
                make_unique<Gaussian>(*this);

            for (auto& parameter : new_distribution->parameters) {
                parameter = parameter->clone();
            }

            return new_distribution;
        }

        string Gaussian::get_description() const {
            stringstream ss;
            const shared_ptr<Node>& mean_node =
                this->parameters[PARAMETER_INDEX::mean];
            const shared_ptr<Node>& variance_node =
                this->parameters[PARAMETER_INDEX::variance];
            ss << "N(" << *mean_node << ", " << *variance_node << ")";

            return ss.str();
        }

        int Gaussian::get_sample_size() const { return 1; }

    } // namespace model
} // namespace tomcat
