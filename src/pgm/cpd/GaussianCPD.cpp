#include "pgm/cpd/GaussianCPD.h"

#include "pgm/ConstantNode.h"
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

        string GaussianCPD::get_name() const {
            return "Gaussian";
        }

        void GaussianCPD::add_to_sufficient_statistics(
            const vector<double>& values) {
            throw invalid_argument("Not implemented yet.");
        }

        Eigen::MatrixXd GaussianCPD::sample_from_conjugacy(
            const shared_ptr<gsl_rng>& random_generator,
            int num_samples,
            const shared_ptr<const RandomVariableNode>& cpd_owner) const {
            throw invalid_argument("Not implemented yet.");
        }

        void GaussianCPD::reset_sufficient_statistics() {
            throw invalid_argument("Not implemented yet.");
        }

        bool GaussianCPD::is_continuous() const {
            return false;
        }

    } // namespace model
} // namespace tomcat
