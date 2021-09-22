#include "InverseGamma.h"

#include <gsl/gsl_cdf.h>
#include <gsl/gsl_randist.h>
#include <math.h>

#include "pgm/NumericNode.h"

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        InverseGamma::InverseGamma(const shared_ptr<Node>& alpha,
                                   const shared_ptr<Node>& beta)
            : Gamma(alpha, beta) {}

        InverseGamma::InverseGamma(shared_ptr<Node>&& alpha,
                                   shared_ptr<Node>&& beta)
            : Gamma(alpha, beta) {}

        InverseGamma::InverseGamma(const vector<shared_ptr<Node>>& parameters)
            : Gamma(parameters) {
            // The vector is here just to maintain the same interface
            // for all distributions, but an InverseGamma distribution cannot
            // have more than two parameters.
            if (parameters.size() > 1) {
                throw TomcatModelException("An InverseGamma distribution must "
                                           "have two parameter nodes.");
            }
        }

        InverseGamma::InverseGamma(vector<shared_ptr<Node>>&& parameters)
            : Gamma(parameters) {
            if (parameters.size() > 1) {
                throw TomcatModelException("An InverseGamma distribution must "
                                           "have two parameter nodes.");
            }
        }

        InverseGamma::InverseGamma(unsigned int alpha, unsigned int beta)
            : Gamma(alpha, beta) {}

        InverseGamma::~InverseGamma() {}

        //----------------------------------------------------------------------
        // Copy & Move constructors/assignments
        //----------------------------------------------------------------------
        InverseGamma::InverseGamma(const InverseGamma& inverse_gamma)
            : Gamma(inverse_gamma.parameters) {
            this->copy(inverse_gamma);
        }

        InverseGamma&
        InverseGamma::operator=(const InverseGamma& inverse_gamma) {
            this->copy(inverse_gamma);
            return *this;
        }

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        Eigen::VectorXd InverseGamma::sample_from_gsl(
            const shared_ptr<gsl_rng>& random_generator,
            double alpha,
            double beta) const {

            double sample =
                1 / gsl_ran_gamma(random_generator.get(), alpha, beta);

            Eigen::VectorXd sample_vector(1);
            sample_vector(0) = sample;

            return sample_vector;
        }

        Eigen::VectorXd InverseGamma::sample_from_conjugacy(
            const shared_ptr<gsl_rng>& random_generator,
            int parameter_idx,
            const Eigen::VectorXd& sufficient_statistics) const {

            int n = sufficient_statistics.size();
            int sum_squares = sufficient_statistics.array().square().sum();

            Eigen::VectorXd parameters = this->get_parameters(parameter_idx);
            double alpha = parameters(PARAMETER_INDEX::alpha);
            double beta = parameters(PARAMETER_INDEX::beta);

            double new_alpha = alpha + n / 2;
            double new_beta = 1 / (1 / beta + sum_squares / 2);

            return this->sample_from_gsl(random_generator, new_alpha, new_beta);
        }

        double InverseGamma::get_pdf(double value) const {
            Eigen::VectorXd parameters = this->get_parameters(0);
            double alpha = parameters(PARAMETER_INDEX::alpha);
            double beta = parameters(PARAMETER_INDEX::beta);

            return (pow(beta, alpha) / tgamma(alpha)) * pow(value, -alpha - 1) *
                   exp(-beta / value);
        }

        double Gamma::get_cdf(double value, bool reverse) const {
            Eigen::VectorXd parameters = this->get_parameters(0);
            double alpha = parameters(PARAMETER_INDEX::alpha);
            double beta = parameters(PARAMETER_INDEX::beta);

            double cdf = gsl_cdf_gamma_Pinv(value, alpha, beta);
            if (reverse) {
                cdf = 1 - cdf;
            }

            return cdf;
        }

        unique_ptr<Distribution> InverseGamma::clone() const {
            unique_ptr<InverseGamma> new_distribution =
                make_unique<InverseGamma>(*this);

            for (auto& parameter : new_distribution->parameters) {
                // Do not clone numeric nodes to allow them to be sharable.
                if (parameter->is_random_variable()) {
                    parameter = parameter->clone();
                }
            }

            return new_distribution;
        }

        string InverseGamma::get_description() const {
            stringstream ss;
            const shared_ptr<Node>& alpha_node =
                this->parameters[PARAMETER_INDEX::alpha];
            const shared_ptr<Node>& beta_node =
                this->parameters[PARAMETER_INDEX::beta];
            ss << "InverseGamma(" << *alpha_node << ", " << *beta_node << ")";

            return ss.str();
        }

    } // namespace model
} // namespace tomcat
