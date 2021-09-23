#include "pgm/cpd/GammaCPD.h"

#include "pgm/NumericNode.h"
#include "pgm/RandomVariableNode.h"

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        GammaCPD::GammaCPD(
            const vector<shared_ptr<NodeMetadata>>& parent_node_order,
            const vector<shared_ptr<Gamma>>& distributions)
            : CPD(parent_node_order) {

            this->distributions.reserve(distributions.size());
            for (const auto& distribution : distributions) {
                this->distributions.push_back(distribution);
            }
            this->sufficient_statistics = Eigen::VectorXd::Zero(2);
        }

        GammaCPD::GammaCPD(vector<shared_ptr<NodeMetadata>>&& parent_node_order,
                           vector<shared_ptr<Gamma>>&& distributions)
            : CPD(parent_node_order) {

            this->distributions.reserve(distributions.size());
            for (const auto& distribution : distributions) {
                this->distributions.push_back(distribution);
            }
            this->sufficient_statistics = Eigen::VectorXd::Zero(2);
        }

        GammaCPD::GammaCPD(
            const vector<shared_ptr<NodeMetadata>>& parent_node_order,
            const Eigen::MatrixXd& parameters)
            : CPD(parent_node_order) {
            this->init_from_matrix(parameters);
        }

        GammaCPD::GammaCPD(vector<shared_ptr<NodeMetadata>>&& parent_node_order,
                           const Eigen::MatrixXd& parameters)
            : CPD(parent_node_order) {
            this->init_from_matrix(parameters);
        }

        GammaCPD::~GammaCPD() {}

        //----------------------------------------------------------------------
        // Copy & Move constructors/assignments
        //----------------------------------------------------------------------
        GammaCPD::GammaCPD(const GammaCPD& cpd) {
            this->copy_cpd(cpd);
            this->sufficient_statistics = cpd.sufficient_statistics;
        }

        GammaCPD& GammaCPD::operator=(const GammaCPD& cpd) {
            this->copy_cpd(cpd);
            this->sufficient_statistics = cpd.sufficient_statistics;
            return *this;
        }

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        void GammaCPD::init_from_matrix(const Eigen::MatrixXd& matrix) {
            for (int row = 0; row < matrix.rows(); row++) {
                for (int i = 0; i < matrix.rows(); i++) {
                    double alpha = matrix(i, Gamma::PARAMETER_INDEX::alpha);
                    double beta = matrix(i, Gamma::PARAMETER_INDEX::beta);
                    shared_ptr<Gamma> distribution_ptr =
                        make_shared<Gamma>(alpha, beta);
                    this->distributions.push_back(distribution_ptr);
                }
            }
            this->sufficient_statistics = Eigen::VectorXd::Zero(2);
            this->freeze_distributions(0);
        }

        unique_ptr<CPD> GammaCPD::clone() const {
            unique_ptr<GammaCPD> new_cpd = make_unique<GammaCPD>(*this);
            new_cpd->clone_distributions();
            new_cpd->sufficient_statistics = this->sufficient_statistics;
            return new_cpd;
        }

        void GammaCPD::clone_distributions() {
            for (auto& distribution : this->distributions) {
                shared_ptr<Distribution> temp = distribution->clone();
                distribution = dynamic_pointer_cast<Gamma>(temp);
            }
        }

        string GammaCPD::get_name() const { return "Gamma"; }

        void GammaCPD::add_to_sufficient_statistics(
            const shared_ptr<const Distribution>& distribution,
            const vector<double>& values) {
            scoped_lock lock(*this->sufficient_statistics_mutex);

            int sum = Eigen::VectorXd::Map(values.data(), values.size())
                          .array()
                          .sum();
            this->sufficient_statistics(0) += sum;
            this->sufficient_statistics(1) += values.size();
        }

        Eigen::MatrixXd GammaCPD::sample_from_conjugacy(
            const shared_ptr<gsl_rng>& random_generator,
            int num_samples,
            const shared_ptr<const RandomVariableNode>& cpd_owner) const {

            // TODO - Fix this if we need a parameter to depend on another node.
            int distribution_idx = 0;
            const auto& prior_distribution =
                this->distributions[distribution_idx];

            const auto& alpha_node =
                prior_distribution
                    ->get_parameters()[Gamma::PARAMETER_INDEX::alpha];

            auto& beta_node =
                prior_distribution
                    ->get_parameters()[Gamma::PARAMETER_INDEX::beta];

            int sum_durations = this->sufficient_statistics(0);
            int num_intervals = this->sufficient_statistics(1);

            alpha_node->increment_assignment(sum_durations);
            double old_beta = beta_node->get_assignment()(0, 0);
            double new_beta = old_beta / (old_beta * num_intervals + 1);
            beta_node->set_assignment(
                Eigen::MatrixXd::Constant(1, 1, new_beta));

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

        void GammaCPD::reset_sufficient_statistics() {
            int distribution_idx = 0;
            const auto& distribution = this->distributions[distribution_idx];
            for (const auto& parameter : distribution->get_parameters()) {
                parameter->pop_assignment();
                parameter->stack_assignment();
            }
            this->sufficient_statistics.fill(0);
        }

        bool GammaCPD::is_continuous() const { return false; }

    } // namespace model
} // namespace tomcat
