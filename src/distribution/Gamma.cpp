#include "Gamma.h"

#include <gsl/gsl_cdf.h>
#include <gsl/gsl_randist.h>

#include "pgm/NumericNode.h"

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        Gamma::Gamma(const shared_ptr<Node>& alpha,
                     const shared_ptr<Node>& beta)
            : Distribution({alpha, beta}) {}

        Gamma::Gamma(shared_ptr<Node>&& alpha, shared_ptr<Node>&& beta)
            : Distribution({move(alpha), move(beta)}) {}

        Gamma::Gamma(const vector<shared_ptr<Node>>& parameters)
            : Distribution(parameters) {
            // The vector is here just to maintain the same interface
            // for all distributions, but a gamma distribution cannot have
            // more than two parameters.
            if (parameters.size() > 1) {
                throw TomcatModelException(
                    "A gamma distribution must have two parameter nodes.");
            }
        }

        Gamma::Gamma(vector<shared_ptr<Node>>&& parameters)
            : Distribution(move(parameters)) {
            if (parameters.size() > 1) {
                throw TomcatModelException(
                    "A gamma distribution must have two parameter nodes.");
            }
        }

        Gamma::Gamma(unsigned int alpha, unsigned int beta) {
            NumericNode alpha_node(alpha);
            NumericNode beta_node(beta);

            this->parameters.push_back(
                make_shared<NumericNode>(move(alpha_node)));
            this->parameters.push_back(
                make_shared<NumericNode>(move(beta_node)));
        }

        Gamma::~Gamma() {}

        //----------------------------------------------------------------------
        // Copy & Move constructors/assignments
        //----------------------------------------------------------------------
        Gamma::Gamma(const Gamma& gamma) { this->copy(gamma); }

        Gamma& Gamma::operator=(const Gamma& gamma) {
            this->copy(gamma);
            return *this;
        }

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        Eigen::VectorXd
        Gamma::sample(const shared_ptr<gsl_rng>& random_generator,
                      int parameter_idx) const {

            Eigen::VectorXd parameters = this->get_parameters(parameter_idx);
            double alpha = parameters(PARAMETER_INDEX::alpha);
            double beta = parameters(PARAMETER_INDEX::beta);

            return this->sample_from_gsl(random_generator, alpha, beta);
        }

        Eigen::VectorXd Gamma::get_parameters(int parameter_idx) const {
            Eigen::VectorXd alphas =
                this->parameters[PARAMETER_INDEX::alpha]->get_assignment().col(
                    0);
            Eigen::VectorXd betas =
                this->parameters[PARAMETER_INDEX::beta]->get_assignment().col(
                    0);

            int alpha_idx = alphas.size() == 1 ? 0 : parameter_idx;
            double alpha = alphas(alpha_idx);

            int beta_idx = betas.size() == 1 ? 0 : parameter_idx;
            double beta = betas(beta_idx);

            Eigen::VectorXd parameters(2);
            parameters(PARAMETER_INDEX::alpha) = alpha;
            parameters(PARAMETER_INDEX::beta) = beta;

            return parameters;
        }

        Eigen::VectorXd
        Gamma::sample_from_gsl(const shared_ptr<gsl_rng>& random_generator,
                               double alpha,
                               double beta) const {

            double sample = gsl_ran_gamma(random_generator.get(), alpha, beta);

            Eigen::VectorXd sample_vector(1);
            sample_vector(0) = sample;

            return sample_vector;
        }

        Eigen::VectorXd
        Gamma::sample(const shared_ptr<gsl_rng>& random_generator,
                      const Eigen::VectorXd& weights) const {

            Eigen::VectorXd parameters = this->get_parameters(0) * weights;
            double alpha = parameters(PARAMETER_INDEX::alpha);
            double beta = parameters(PARAMETER_INDEX::beta);

            return this->sample_from_gsl(random_generator, alpha, beta);
        }

        Eigen::VectorXd
        Gamma::sample(const std::shared_ptr<gsl_rng>& random_generator,
                      const Eigen::VectorXd& weights,
                      double replace_by_weight) const {
            throw TomcatModelException("Not defined for continuous "
                                       "distributions.");
        }

        Eigen::VectorXd Gamma::sample_from_conjugacy(
            const shared_ptr<gsl_rng>& random_generator,
            int parameter_idx,
            const Eigen::VectorXd& sufficient_statistics) const {

            int sum_durations = sufficient_statistics(0);
            int num_intervals = sufficient_statistics(1);

            Eigen::VectorXd parameters = this->get_parameters(parameter_idx);
            double alpha = parameters(PARAMETER_INDEX::alpha);
            double beta = parameters(PARAMETER_INDEX::beta);

            double new_alpha = alpha + sum_durations;
            double new_beta = beta / (beta * num_intervals + 1);

            return this->sample_from_gsl(random_generator, new_alpha, new_beta);
        }

        double Gamma::get_pdf(const Eigen::VectorXd& value) const {
            return this->get_pdf(value(0));
        }

        double Gamma::get_pdf(double value) const {
            Eigen::VectorXd parameters = this->get_parameters(0);
            double alpha = parameters(PARAMETER_INDEX::alpha);
            double beta = parameters(PARAMETER_INDEX::beta);

            return gsl_ran_gamma_pdf(value, alpha, beta);
        }

        double Gamma::get_cdf(double value, bool reverse) const {
            Eigen::VectorXd parameters = this->get_parameters(0);
            double alpha = parameters(PARAMETER_INDEX::alpha);
            double beta = parameters(PARAMETER_INDEX::beta);

            double cdf = gsl_cdf_gamma_P(value, alpha, beta);
            if (reverse) {
                cdf = 1 - cdf;
            }

            return cdf;
        }

        unique_ptr<Distribution> Gamma::clone() const {
            unique_ptr<Gamma> new_distribution = make_unique<Gamma>(*this);

            for (auto& parameter : new_distribution->parameters) {
                // Do not clone numeric nodes to allow them to be sharable.
                if (parameter->is_random_variable()) {
                    parameter = parameter->clone();
                }
            }

            return new_distribution;
        }

        string Gamma::get_description() const {
            stringstream ss;
            const shared_ptr<Node>& alpha_node =
                this->parameters[PARAMETER_INDEX::alpha];
            const shared_ptr<Node>& beta_node =
                this->parameters[PARAMETER_INDEX::beta];
            ss << "Gamma(" << *alpha_node << ", " << *beta_node << ")";

            return ss.str();
        }

        int Gamma::get_sample_size() const { return 1; }

        void
        Gamma::update_from_posterior(const Eigen::VectorXd& posterior_weights) {
            // Not implemented
        }

    } // namespace model
} // namespace tomcat
