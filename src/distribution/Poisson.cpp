#include "Poisson.h"

#include <gsl/gsl_randist.h>

#include "pgm/ConstantNode.h"
#include "pgm/RandomVariableNode.h"

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        Poisson::Poisson(const shared_ptr<Node>& lambda)
            : Distribution({lambda}) {}

        Poisson::Poisson(shared_ptr<Node>&& lambda)
            : Distribution({move(lambda)}) {}

        Poisson::Poisson(const unsigned int lambda) {
            shared_ptr<ConstantNode> lambda_node =
                make_shared<ConstantNode>(ConstantNode(lambda));
            this->parameters.push_back(move(lambda_node));
        }

        Poisson::~Poisson() {}

        //----------------------------------------------------------------------
        // Copy & Move constructors/assignments
        //----------------------------------------------------------------------
        Poisson::Poisson(const Poisson& poisson) {
            this->copy(poisson);
        }

        Poisson& Poisson::operator=(const Poisson& poisson) {
            this->copy(poisson);
            return *this;
        }

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        Eigen::VectorXd
        Poisson::sample(const shared_ptr<gsl_rng>& random_generator,
                        int parameter_idx) const {
            Eigen::MatrixXd lambdas = this->parameters[0]->get_assignment();

            parameter_idx = lambdas.rows() == 1 ? 0 : parameter_idx;

            return this->sample_from_gsl(random_generator,
                                         lambdas(parameter_idx, 0));
        }

        Eigen::VectorXd
        Poisson::sample_from_gsl(const shared_ptr<gsl_rng>& random_generator,
                                 unsigned int lambda) const {

            unsigned int value =
                gsl_ran_poisson(random_generator.get(), lambda);

            return Eigen::VectorXd::Constant(1, value);
        }

        Eigen::VectorXd
        Poisson::sample(const shared_ptr<gsl_rng>& random_generator,
                        const Eigen::VectorXd& weight) const {

            // Parameter nodes are never in-plate. Therefore, their
            // assignment matrix is always comprised by a single row.
            unsigned int lambda = this->parameters[0]->get_assignment()(0, 0);

            lambda = lambda * weight(0);

            return this->sample_from_gsl(random_generator, lambda);
        }

        Eigen::VectorXd Poisson::sample_from_conjugacy(
            const shared_ptr<gsl_rng>& random_generator,
            int parameter_idx,
            const Eigen::VectorXd& sufficient_statistics) const {
            throw invalid_argument(
                "No conjugate prior with a Poisson distribution.");
        }

        double Poisson::get_pdf(const Eigen::VectorXd& value) const {
            unsigned int lambda = this->parameters[0]->get_assignment()(0, 0);
            return gsl_ran_poisson_pdf(value(0), lambda);
        }

        unique_ptr<Distribution> Poisson::clone() const {
            unique_ptr<Poisson> new_distribution = make_unique<Poisson>(*this);
            new_distribution->parameters[0] = new_distribution->parameters[0]->clone();

            return new_distribution;
        }

        string Poisson::get_description() const {
            stringstream ss;
            ss << "Poisson(" << *this->parameters[0] << ")";

            return ss.str();
        }

        int Poisson::get_sample_size() const { return 1; }

    } // namespace model
} // namespace tomcat
