#include "Categorical.h"

#include <gsl/gsl_randist.h>

#include "pgm/NumericNode.h"
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

        Categorical::Categorical(const vector<shared_ptr<Node>>& probabilities)
            : Distribution(probabilities) {
            if (probabilities.size() > 1) {
                // Merge individual nodes in a single node containing all the
                // probabilities
                Eigen::VectorXd parameter_values(probabilities.size());
                int i = 0;
                for (const auto& probability_node : probabilities) {
                    parameter_values(i++) =
                        probability_node->get_assignment()(0, 0);
                }
                NumNodePtr parameter =
                    make_shared<NumericNode>(parameter_values);
                this->parameters = {parameter};
            }
        }

        Categorical::Categorical(vector<shared_ptr<Node>>&& probabilities)
            : Distribution(move(probabilities)) {
            if (probabilities.size() > 1) {
                // Merge individual nodes in a single node containing all the
                // probabilities
                Eigen::MatrixXd parameter_values(1, probabilities.size());
                int col = 0;
                for (const auto& probability_node : probabilities) {
                    parameter_values(0, col++) =
                        probability_node->get_assignment()(0, 0);
                }
                NumNodePtr parameter =
                    make_shared<NumericNode>(parameter_values);
                this->parameters = {parameter};
            }
        }

        Categorical::Categorical(const Eigen::VectorXd& probabilities) {
            shared_ptr<NumericNode> probabilities_node =
                make_shared<NumericNode>(NumericNode(probabilities));
            this->parameters.push_back(move(probabilities_node));
        }

        Categorical::Categorical(const Eigen::VectorXd&& probabilities) {
            shared_ptr<NumericNode> probabilities_node =
                make_shared<NumericNode>(NumericNode(move(probabilities)));
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

        Eigen::VectorXd
        Categorical::sample(const std::shared_ptr<gsl_rng>& random_generator,
                            const Eigen::VectorXd& weights,
                            double replace_by_weight) const {

            Eigen::VectorXd probabilities =
                this->parameters[0]->get_assignment().row(0);
            probabilities[(int)replace_by_weight] = 1;

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
            return this->get_pdf(value(0));
        }

        double Categorical::get_pdf(double value) const {
            Eigen::VectorXd probabilities =
                this->parameters[0]->get_assignment().row(0);

            return probabilities((int)value);
        }

        double Categorical::get_cdf(double value, bool reverse) const {
            Eigen::VectorXd probabilities =
                this->parameters[0]->get_assignment().row(0);

            double cdf = probabilities.block(0, 0, (int)value + 1, 1).sum();
            if (reverse) {
                cdf = 1 - cdf;
            }

            return cdf;
        }

        unique_ptr<Distribution> Categorical::clone() const {
            unique_ptr<Categorical> new_distribution =
                make_unique<Categorical>(*this);

            for (auto& parameter : new_distribution->parameters) {
                // Do not clone numeric nodes to allow them to be sharable.
                if (dynamic_pointer_cast<RandomVariableNode>(parameter)) {
                    parameter = parameter->clone();
                }
            }
            return new_distribution;
        }

        string Categorical::get_description() const {
            stringstream ss;
            ss << "Cat(" << *this->parameters[0] << ")";

            return ss.str();
        }

        int Categorical::get_sample_size() const { return 1; }

        void Categorical::update_from_posterior(
            const Eigen::VectorXd& posterior_weights) {

            Eigen::VectorXd weighted_probabilities = this->parameters[0]
                                                         ->get_assignment()
                                                         .row(0)
                                                         .transpose()
                                                         .array() *
                                                     posterior_weights.array();

            if (weighted_probabilities.sum() < EPSILON) {
                weighted_probabilities = Eigen::VectorXd::Ones(
                    this->parameters[0]->get_assignment().cols());
            }

            weighted_probabilities /= weighted_probabilities.sum();

            this->parameters[0] =
                make_shared<NumericNode>(weighted_probabilities);
        }

    } // namespace model
} // namespace tomcat
