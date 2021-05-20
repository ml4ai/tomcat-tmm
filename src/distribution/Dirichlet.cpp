#include "Dirichlet.h"

#include <gsl/gsl_randist.h>

#include "pgm/ConstantNode.h"
#include "pgm/RandomVariableNode.h"

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        Dirichlet::Dirichlet(const vector<shared_ptr<Node>>& alpha)
            : Distribution(alpha) {
            this->init_constant_alpha();
        }

        Dirichlet::Dirichlet(vector<shared_ptr<Node>>&& alpha)
            : Distribution(move(alpha)) {
            this->init_constant_alpha();
        }

        Dirichlet::Dirichlet(const Eigen::VectorXd& alpha) {
            this->constant_alpha = Eigen::MatrixXd(1, alpha.size());
            this->constant_alpha.row(0) = alpha;

            for (int i = 0; i < alpha.size(); i++) {
                ConstantNode parameter_node(alpha(i));
                this->parameters.push_back(
                    make_shared<ConstantNode>(move(parameter_node)));
            }
        }

        Dirichlet::~Dirichlet() {}

        //----------------------------------------------------------------------
        // Copy & Move constructors/assignments
        //----------------------------------------------------------------------
        Dirichlet::Dirichlet(const Dirichlet& dirichlet) {
            this->copy(dirichlet);
            this->constant_alpha = dirichlet.constant_alpha;
        }

        Dirichlet& Dirichlet::operator=(const Dirichlet& dirichlet) {
            this->copy(dirichlet);
            this->constant_alpha = dirichlet.constant_alpha;
            return *this;
        }

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        void Dirichlet::init_constant_alpha() {
            for (auto& parameter : this->parameters) {
                if (!dynamic_pointer_cast<ConstantNode>(parameter)) {
                    // Alpha can only be constant if all the parameter nodes
                    // that composes it are constant.
                    return;
                }
            }

            int rows = this->parameters[0]->get_assignment().rows();
            int cols = this->parameters.size();
            this->constant_alpha = Eigen::MatrixXd(rows, cols);
            for (int i = 0; i < cols; i++) {
                this->constant_alpha.col(i) =
                    this->parameters[i]->get_assignment().col(0);
            }
        }

        Eigen::VectorXd
        Dirichlet::sample(const shared_ptr<gsl_rng>& random_generator,
                          int parameter_idx) const {
            Eigen::VectorXd alpha = this->get_alpha(parameter_idx);

            return this->sample_from_gsl(random_generator, alpha);
        }

        Eigen::VectorXd Dirichlet::get_alpha(int parameter_idx) const {
            if (this->constant_alpha.size() > 0) {
                parameter_idx =
                    this->constant_alpha.rows() == 1 ? 0 : parameter_idx;
                return this->constant_alpha.row(parameter_idx);
            }
            else {
                Eigen::VectorXd alpha(this->parameters.size());

                for (int i = 0; i < alpha.size(); i++) {
                    int rows = this->parameters[i]->get_assignment().rows();
                    parameter_idx = rows == 1 ? 0 : parameter_idx;
                    alpha(i) =
                        this->parameters[i]->get_assignment()(parameter_idx, 0);
                }

                return alpha;
            }
        }

        Eigen::VectorXd
        Dirichlet::sample_from_gsl(const shared_ptr<gsl_rng>& random_generator,
                                   const Eigen::VectorXd& parameters) const {
            int k = this->parameters.size();
            double* sample_ptr = new double[k];

            const double* alpha = parameters.data();

            gsl_ran_dirichlet(random_generator.get(), k, alpha, sample_ptr);

            Eigen::Map<Eigen::VectorXd> sample(sample_ptr, k);

            return sample;
        }

        Eigen::VectorXd
        Dirichlet::sample(const shared_ptr<gsl_rng>& random_generator,
                          const Eigen::VectorXd& weights) const {
            Eigen::VectorXd alpha = this->get_alpha(0) * weights;

            return this->sample_from_gsl(random_generator, alpha);
        }

        Eigen::VectorXd
        Dirichlet::sample(const std::shared_ptr<gsl_rng>& random_generator,
                          const Eigen::VectorXd& weights,
                          double replace_by_weight) const {
            throw TomcatModelException("Not defined for continuous "
                                       "distributions.");
        }

        Eigen::VectorXd Dirichlet::sample_from_conjugacy(
            const shared_ptr<gsl_rng>& random_generator,
            int parameter_idx,
            const Eigen::VectorXd& sufficient_statistics) const {

            Eigen::VectorXd alpha = this->get_alpha(parameter_idx);
            alpha = alpha + sufficient_statistics;

            return this->sample_from_gsl(random_generator, alpha);
        }

        double Dirichlet::get_pdf(const Eigen::VectorXd& value) const {
            Eigen::VectorXd alpha = this->get_alpha(0);
            int k = alpha.size();
            const double* alpha_ptr = alpha.data();
            const double* value_ptr = value.data();

            return gsl_ran_dirichlet_pdf(k, alpha_ptr, value_ptr);
        }

        double Dirichlet::get_pdf(double value) const {
            throw TomcatModelException("The assignment of a dirichlet "
                                       "distribution must be a vector.");
        }

        double Dirichlet::get_cdf(double value, bool reverse) const {
            throw TomcatModelException("No CDF for Dirichlet distribution.");
        }

        unique_ptr<Distribution> Dirichlet::clone() const {
            unique_ptr<Dirichlet> new_distribution =
                make_unique<Dirichlet>(*this);

            for (auto& parameter : new_distribution->parameters) {
                parameter = parameter->clone();
            }

            return new_distribution;
        }

        string Dirichlet::get_description() const {
            stringstream ss;
            ss << "Dir\n";
            ss << "(\n";
            for (const auto& parameter : this->parameters) {
                ss << " " << *parameter << "\n";
            }
            ss << ")";

            return ss.str();
        }

        int Dirichlet::get_sample_size() const {
            return this->parameters.size();
        }

    } // namespace model
} // namespace tomcat
