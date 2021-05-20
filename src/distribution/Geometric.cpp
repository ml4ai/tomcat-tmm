#include "Geometric.h"

#include <gsl/gsl_cdf.h>
#include <gsl/gsl_randist.h>

#include "pgm/ConstantNode.h"
#include "pgm/RandomVariableNode.h"

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        Geometric::Geometric(const shared_ptr<Node>& p) : Distribution({p}) {}

        Geometric::Geometric(shared_ptr<Node>&& p) : Distribution({move(p)}) {}

        Geometric::Geometric(double p) {
            shared_ptr<ConstantNode> p_node =
                make_shared<ConstantNode>(ConstantNode(p));
            this->parameters.push_back(move(p_node));
        }

        Geometric::~Geometric() {}

        //----------------------------------------------------------------------
        // Copy & Move constructors/assignments
        //----------------------------------------------------------------------
        Geometric::Geometric(const Geometric& geometric) {
            this->copy(geometric);
        }

        Geometric& Geometric::operator=(const Geometric& geometric) {
            this->copy(geometric);
            return *this;
        }

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        Eigen::VectorXd
        Geometric::sample(const shared_ptr<gsl_rng>& random_generator,
                          int parameter_idx) const {
            Eigen::MatrixXd ps = this->parameters[0]->get_assignment();

            parameter_idx = ps.rows() == 1 ? 0 : parameter_idx;
            double p = ps(parameter_idx, 0);

            return this->sample_from_gsl(random_generator, p);
        }

        Eigen::VectorXd
        Geometric::sample_from_gsl(const shared_ptr<gsl_rng>& random_generator,
                                   double p) const {

            unsigned int value = gsl_ran_geometric(random_generator.get(), p);

            return Eigen::VectorXd::Constant(1, value);
        }

        Eigen::VectorXd
        Geometric::sample(const shared_ptr<gsl_rng>& random_generator,
                          const Eigen::VectorXd& weight) const {

            // Parameter nodes are never in-plate. Therefore, their
            // assignment matrix is always comprised by a single row.
            double p = this->parameters[0]->get_assignment()(0, 0);
            p = p * weight(0);

            return this->sample_from_gsl(random_generator, p);
        }

        Eigen::VectorXd
        Geometric::sample(const std::shared_ptr<gsl_rng>& random_generator,
                          const Eigen::VectorXd& weights,
                          double replace_by_weight) const {
            throw TomcatModelException("Not implemented for Geometric "
                                       "distribution.");
        }

        Eigen::VectorXd Geometric::sample_from_conjugacy(
            const shared_ptr<gsl_rng>& random_generator,
            int parameter_idx,
            const Eigen::VectorXd& sufficient_statistics) const {
            throw invalid_argument(
                "No conjugate prior with a Geometric distribution.");
        }

        double Geometric::get_pdf(const Eigen::VectorXd& value) const {
            return this->get_pdf(value(0));
        }

        double Geometric::get_pdf(double value) const {
            double p = this->parameters[0]->get_assignment()(0, 0);
            double pdf = gsl_ran_geometric_pdf(value + 1, p);
            return pdf;
        }

        double Geometric::get_cdf(double value, bool reverse) const {
            double p = this->parameters[0]->get_assignment()(0, 0);

            double cdf =
                (value < 0) ? 0 : gsl_cdf_geometric_P((int)value + 1, p);
            if (reverse) {
                cdf = 1 - cdf;
            }

            return cdf;
        }

        unique_ptr<Distribution> Geometric::clone() const {
            unique_ptr<Geometric> new_distribution =
                make_unique<Geometric>(*this);
            new_distribution->parameters[0] =
                new_distribution->parameters[0]->clone();

            return new_distribution;
        }

        string Geometric::get_description() const {
            stringstream ss;
            ss << "Geometric(" << *this->parameters[0] << ")";

            return ss.str();
        }

        int Geometric::get_sample_size() const { return 1; }

    } // namespace model
} // namespace tomcat
