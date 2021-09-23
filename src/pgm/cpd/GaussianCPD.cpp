#include "pgm/cpd/GaussianCPD.h"

#include "pgm/NumericNode.h"
#include "pgm/RandomVariableNode.h"

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        GaussianCPD::GaussianCPD(
            const vector<shared_ptr<NodeMetadata>>& parent_node_order,
            const vector<shared_ptr<Gaussian>>& distributions)
            : CPD(parent_node_order) {

            this->distributions.reserve(distributions.size());
            for (const auto& distribution : distributions) {
                this->distributions.push_back(distribution);
            }
        }

        GaussianCPD::GaussianCPD(
            vector<shared_ptr<NodeMetadata>>&& parent_node_order,
            vector<shared_ptr<Gaussian>>&& distributions)
            : CPD(parent_node_order) {

            this->distributions.reserve(distributions.size());
            for (const auto& distribution : distributions) {
                this->distributions.push_back(distribution);
            }
        }

        GaussianCPD::GaussianCPD(
            const vector<shared_ptr<NodeMetadata>>& parent_node_order,
            const Eigen::MatrixXd& parameters)
            : CPD(parent_node_order) {
            this->init_from_matrix(parameters);
        }

        GaussianCPD::GaussianCPD(
            vector<shared_ptr<NodeMetadata>>&& parent_node_order,
            const Eigen::MatrixXd& parameters)
            : CPD(parent_node_order) {
            this->init_from_matrix(parameters);
        }

        GaussianCPD::~GaussianCPD() {}

        //----------------------------------------------------------------------
        // Copy & Move constructors/assignments
        //----------------------------------------------------------------------
        GaussianCPD::GaussianCPD(const GaussianCPD& cpd) {
            this->copy_cpd(cpd);
        }

        GaussianCPD& GaussianCPD::operator=(const GaussianCPD& cpd) {
            this->copy_cpd(cpd);
            return *this;
        }

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        void GaussianCPD::init_from_matrix(const Eigen::MatrixXd& matrix) {
            for (int row = 0; row < matrix.rows(); row++) {
                for (int i = 0; i < matrix.rows(); i++) {
                    double mean = matrix(i, Gaussian::PARAMETER_INDEX::mean);
                    double variance =
                        matrix(i, Gaussian::PARAMETER_INDEX::variance);
                    shared_ptr<Gaussian> distribution_ptr =
                        make_shared<Gaussian>(mean, variance);
                    this->distributions.push_back(distribution_ptr);
                }
            }
            this->freeze_distributions(0);
        }

        unique_ptr<CPD> GaussianCPD::clone() const {
            unique_ptr<GaussianCPD> new_cpd = make_unique<GaussianCPD>(*this);
            new_cpd->clone_distributions();
            return new_cpd;
        }

        void GaussianCPD::clone_distributions() {
            for (auto& distribution : this->distributions) {
                shared_ptr<Distribution> temp = distribution->clone();
                distribution = dynamic_pointer_cast<Gaussian>(temp);
            }
        }

        string GaussianCPD::get_name() const { return "Gaussian"; }

        void GaussianCPD::add_to_sufficient_statistics(
            const std::shared_ptr<const Distribution>& distribution,
            const vector<double>& values) {

            scoped_lock lock(*this->sufficient_statistics_mutex);

            this->variance_for_posterior =
                distribution
                    ->get_parameters()[Gaussian::PARAMETER_INDEX::variance]
                    ->get_assignment()(0, 0);
            this->sufficient_statistics.insert(
                this->sufficient_statistics.end(),
                values.begin(),
                values.end());
        }

        Eigen::MatrixXd GaussianCPD::sample_from_conjugacy(
            const shared_ptr<gsl_rng>& random_generator,
            int num_samples,
            const shared_ptr<const RandomVariableNode>& cpd_owner) const {

            // TODO - Fix this if we need a parameter to depend on another node.
            int distribution_idx = 0;
            const auto& prior_distribution =
                this->distributions[distribution_idx];

            int n = this->sufficient_statistics.size();
            int sum =
                Eigen::VectorXd::Map(this->sufficient_statistics.data(), n)
                    .array()
                    .sum();

            const auto& mean_node =
                prior_distribution
                    ->get_parameters()[Gaussian::PARAMETER_INDEX::mean];
            const auto& variance_node =
                prior_distribution
                    ->get_parameters()[Gaussian::PARAMETER_INDEX::variance];

            double mean_prior = mean_node->get_assignment()(0, 0);
            double variance_prior = variance_node->get_assignment()(0, 0);

            double new_mean =
                (1 / (pow(variance_prior, -2) +
                      n * pow(this->variance_for_posterior, -2))) *
                (mean_prior * pow(variance_prior, -2) +
                 sum * pow(this->variance_for_posterior, -2));
            double new_variance =
                1 / (pow(variance_prior, -2) +
                     n * pow(this->variance_for_posterior, 2));

            mean_node->set_assignment(
                Eigen::MatrixXd::Constant(1, 1, new_mean));
            variance_node->set_assignment(
                Eigen::MatrixXd::Constant(1, 1, new_variance));

            // TODO - remove this later as prior of parameter nodes do not
            //  depend on other nodes for now.
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

        void GaussianCPD::reset_sufficient_statistics() {
            int distribution_idx = 0;
            const auto& distribution = this->distributions[distribution_idx];
            for (const auto& parameter : distribution->get_parameters()) {
                parameter->pop_assignment();
                parameter->stack_assignment();
            }
            this->sufficient_statistics.clear();
        }

        bool GaussianCPD::is_continuous() const { return false; }

        void GaussianCPD::update_sufficient_statistics(
            const shared_ptr<RandomVariableNode>& cpd_owner) {

            int data_size = cpd_owner->get_size();
            Eigen::VectorXi distribution_indices =
                this->get_indexed_distribution_indices(cpd_owner->get_parents(),
                                                       data_size);
            unordered_map<int, vector<double>> values_per_distribution;

            // No need to check if the CPD owner is a timer because a timer
            // cannot have a gaussian distribution since this implementation
            // only works with discrete time.
            const Eigen::MatrixXd& values = cpd_owner->get_assignment();

            for (int i = 0; i < data_size; i++) {
                int distribution_idx = distribution_indices[i];

                // TODO - this needs to be adapted for multivariate gaussian
                double value = values(i, 0);
                if (value != NO_OBS) {
                    values_per_distribution[distribution_idx].push_back(value);
                }
            }

            for (auto& [distribution_idx, values] : values_per_distribution) {
                const shared_ptr<Distribution>& distribution =
                    this->distributions[distribution_idx];
                distribution->update_sufficient_statistics(values);
            }
        }

    } // namespace model
} // namespace tomcat
