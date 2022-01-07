#include "pgm/cpd/InverseGammaCPD.h"

#include "pgm/NumericNode.h"
#include "pgm/RandomVariableNode.h"
#include "distribution/Gaussian.h"

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        InverseGammaCPD::InverseGammaCPD(
            const vector<shared_ptr<NodeMetadata>>& parent_node_order,
            const vector<shared_ptr<InverseGamma>>& distributions)
            : CPD(parent_node_order) {

            this->init_from_distributions(distributions);
        }

        InverseGammaCPD::InverseGammaCPD(
            vector<shared_ptr<NodeMetadata>>&& parent_node_order,
            const vector<shared_ptr<InverseGamma>>& distributions)
            : CPD(parent_node_order) {

            this->init_from_distributions(distributions);
        }

        InverseGammaCPD::InverseGammaCPD(
            const vector<shared_ptr<NodeMetadata>>& parent_node_order,
            const Eigen::MatrixXd& parameters)
            : CPD(parent_node_order) {
            this->init_from_matrix(parameters);
        }

        InverseGammaCPD::InverseGammaCPD(
            vector<shared_ptr<NodeMetadata>>&& parent_node_order,
            const Eigen::MatrixXd& parameters)
            : CPD(parent_node_order) {
            this->init_from_matrix(parameters);
        }

        InverseGammaCPD::~InverseGammaCPD() {}

        //----------------------------------------------------------------------
        // Copy & Move constructors/assignments
        //----------------------------------------------------------------------
        InverseGammaCPD::InverseGammaCPD(const InverseGammaCPD& cpd) {
            this->copy_cpd(cpd);
        }

        InverseGammaCPD&
        InverseGammaCPD::operator=(const InverseGammaCPD& cpd) {
            this->copy_cpd(cpd);
            return *this;
        }

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        void InverseGammaCPD::init_from_distributions(
            const vector<shared_ptr<InverseGamma>>& distributions) {
            this->distributions.reserve(distributions.size());
            for (const auto& distribution : distributions) {
                this->distributions.push_back(distribution);
            }
        }

        void InverseGammaCPD::init_from_matrix(const Eigen::MatrixXd& matrix) {
            for (int row = 0; row < matrix.rows(); row++) {
                for (int i = 0; i < matrix.rows(); i++) {
                    double alpha = matrix(i, Gamma::PARAMETER_INDEX::alpha);
                    double beta = matrix(i, Gamma::PARAMETER_INDEX::beta);
                    shared_ptr<InverseGamma> distribution_ptr =
                        make_shared<InverseGamma>(alpha, beta);
                    this->distributions.push_back(distribution_ptr);
                }
            }
            this->freeze_distributions(0);
        }

        unique_ptr<CPD> InverseGammaCPD::clone() const {
            unique_ptr<InverseGammaCPD> new_cpd =
                make_unique<InverseGammaCPD>(*this);
            new_cpd->clone_distributions();
            return new_cpd;
        }

        void InverseGammaCPD::clone_distributions() {
            for (auto& distribution : this->distributions) {
                shared_ptr<Distribution> temp = distribution->clone();
                distribution = dynamic_pointer_cast<InverseGamma>(temp);
            }
        }

        string InverseGammaCPD::get_name() const { return "InverseGamma"; }

        void InverseGammaCPD::add_to_sufficient_statistics(
            const shared_ptr<const Distribution>& distribution,
            const vector<double>& values) {

            // The InverseGamma as a conjugate prior of a Gaussian distribution
            // with known mean, has a closed-form posterior. Given by another
            // InverseGamma with the parameters defined as below.

            // TODO - Fix this if we need a parameter to depend on another node.
            int distribution_idx = 0;
            const auto& prior_distribution =
                this->distributions[distribution_idx];
            scoped_lock lock(*this->sufficient_statistics_mutex);

            auto alpha_node =
                prior_distribution
                    ->get_parameters()[InverseGamma::PARAMETER_INDEX::alpha];

            auto beta_node =
                prior_distribution
                    ->get_parameters()[InverseGamma::PARAMETER_INDEX::beta];

            double beta_prior = beta_node->get_assignment()(0, 0);
            // An inverse gamma is a conjugate prior of a Gaussian
            double mean =
                distribution->get_parameters()[Gaussian::PARAMETER_INDEX::mean]
                    ->get_assignment()(0, 0);
            double squared_sum =
                (Eigen::VectorXd::Map(values.data(), values.size()).array() -
                 mean)
                    .square()
                    .sum();

            alpha_node->increment_assignment(values.size() / 2.0);
            beta_node->invert_assignment();
            beta_node->increment_assignment(squared_sum / 2);
            beta_node->invert_assignment();
        }

        Eigen::MatrixXd InverseGammaCPD::sample_from_conjugacy(
            const shared_ptr<gsl_rng>& random_generator,
            int num_samples,
            const shared_ptr<const RandomVariableNode>& cpd_owner) const {

            Eigen::VectorXi distribution_indices =
                this->get_indexed_distribution_indices(cpd_owner->get_parents(),
                                                       num_samples);

            int sample_size = this->distributions[0]->get_sample_size();

            Eigen::MatrixXd samples(distribution_indices.size(), sample_size);
            for (int i = 0; i < distribution_indices.size(); i++) {
                int distribution_idx = distribution_indices(i);
                const auto& distribution =
                    this->distributions[distribution_idx];
                Eigen::VectorXd assignment =
                    distribution->sample(random_generator);
                samples.row(i) = move(assignment);
            }

            return samples;
        }

        void InverseGammaCPD::reset_sufficient_statistics() {
            int distribution_idx = 0;
            const auto& distribution = this->distributions[distribution_idx];
            for (const auto& parameter : distribution->get_parameters()) {
                parameter->pop_assignment();
                parameter->stack_assignment();
            }
        }

        bool InverseGammaCPD::is_continuous() const { return true; }

    } // namespace model
} // namespace tomcat
