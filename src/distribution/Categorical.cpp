#include "Categorical.h"

#include <gsl/gsl_randist.h>

#include "pgm/ConstantNode.h"
#include "pgm/RandomVariableNode.h"

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        Categorical::Categorical(const shared_ptr<Node>& probabilities)
            : Distribution({probabilities}) {}

        Categorical::Categorical(shared_ptr<Node>&& probabilities)
            : Distribution({move(probabilities)}) {}

        Categorical::Categorical(const Eigen::VectorXd& probabilities) {
            shared_ptr<ConstantNode> probabilities_node =
                make_shared<ConstantNode>(ConstantNode(probabilities));
            this->parameters.push_back(move(probabilities_node));
        }

        Categorical::Categorical(const Eigen::VectorXd&& probabilities) {
            shared_ptr<ConstantNode> probabilities_node =
                make_shared<ConstantNode>(ConstantNode(move(probabilities)));
            this->parameters.push_back(move(probabilities_node));
        }

        Categorical::~Categorical() {}

        //----------------------------------------------------------------------
        // Copy & Move constructors/assignments
        //----------------------------------------------------------------------
        Categorical::Categorical(const Categorical& categorical) {
            this->copy(categorical);
        }

        Categorical& Categorical::operator=(const Categorical& categorical) {
            this->copy(categorical);
            return *this;
        }

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        Eigen::VectorXd
        Categorical::sample(const shared_ptr<gsl_rng>& random_generator,
                            int parameter_idx) const {
            Eigen::MatrixXd probabilities =
                this->parameters[0]->get_assignment();

            parameter_idx = probabilities.rows() == 1 ? 0 : parameter_idx;

            return this->sample_from_gsl(random_generator,
                                         probabilities.row(parameter_idx));
        }

        Eigen::VectorXd Categorical::sample_from_gsl(
            const shared_ptr<gsl_rng>& random_generator,
            const Eigen::VectorXd& parameters) const {

            Eigen::VectorXd checked_parameters;
            // If for some reason, all the probabilities are zero, sample from a
            // uniform distribution;
            if (parameters.sum() < EPSILON) {
                checked_parameters = Eigen::VectorXd::Ones(parameters.size());
            }
            else {
                checked_parameters = parameters;
            }

            int k = parameters.size();
            const double* parameters_ptr = checked_parameters.data();
            unsigned int* sample_ptr = new unsigned int[k];

            gsl_ran_multinomial(
                random_generator.get(), k, 1, parameters_ptr, sample_ptr);

            Eigen::VectorXd sample_vector(1);
            sample_vector(0) = this->get_sample_index(sample_ptr, k);

            delete[] sample_ptr;

            return sample_vector;
        }

        unsigned int
        Categorical::get_sample_index(const unsigned int* sample_array,
                                      size_t array_size) const {
            return distance(sample_array,
                            find(sample_array, sample_array + array_size, 1));
        }

        Eigen::VectorXd
        Categorical::sample(const shared_ptr<gsl_rng>& random_generator,
                            const Eigen::VectorXd& weights) const {

            // Parameter nodes are never in-plate. Therefore, their
            // assignment matrix is always comprised by a single row.
            Eigen::VectorXd probabilities =
                this->parameters[0]->get_assignment().row(0);

            Eigen::VectorXd weighted_probabilities =
                probabilities.array() * weights.array();

            // The weighted probabilities do not need to be normalized
            // because GSL already does that.
            return this->sample_from_gsl(random_generator,
                                         weighted_probabilities);
        }

        Eigen::VectorXd Categorical::sample_from_conjugacy(
            const shared_ptr<gsl_rng>& random_generator,
            int parameter_idx,
            const Eigen::VectorXd& sufficient_statistics) const {
            throw invalid_argument(
                "No conjugate prior with a categorical distribution.");
        }

        double Categorical::get_pdf(const Eigen::VectorXd& value) const {
            Eigen::VectorXd probabilities =
                this->parameters[0]->get_assignment().row(0);

            return probabilities((int)value(0));
        }

        unique_ptr<Distribution> Categorical::clone() const {
            unique_ptr<Categorical> new_distribution =
                make_unique<Categorical>(*this);
            new_distribution->parameters[0] =
                new_distribution->parameters[0]->clone();

            return new_distribution;
        }

        string Categorical::get_description() const {
            stringstream ss;
            ss << "Cat(" << *this->parameters[0] << ")";

            return ss.str();
        }

        int Categorical::get_sample_size() const { return 1; }

    } // namespace model
} // namespace tomcat
