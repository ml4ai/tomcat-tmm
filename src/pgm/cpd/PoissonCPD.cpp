#include "pgm/cpd/PoissonCPD.h"

#include "pgm/NumericNode.h"
#include "pgm/RandomVariableNode.h"

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        PoissonCPD::PoissonCPD(
            const vector<shared_ptr<NodeMetadata>>& parent_node_order,
            const vector<shared_ptr<Poisson>>& distributions)
            : CPD(parent_node_order) {

            this->init_from_distributions(distributions);
        }

        PoissonCPD::PoissonCPD(
            vector<shared_ptr<NodeMetadata>>&& parent_node_order,
            const vector<shared_ptr<Poisson>>& distributions)
            : CPD(parent_node_order) {

            this->init_from_distributions(distributions);
        }

        PoissonCPD::PoissonCPD(
            const vector<shared_ptr<NodeMetadata>>& parent_node_order,
            const Eigen::VectorXd& lambdas)
            : CPD(parent_node_order) {
            this->init_from_vector(lambdas);
        }

        PoissonCPD::PoissonCPD(
            vector<shared_ptr<NodeMetadata>>&& parent_node_order,
            const Eigen::VectorXd& lambdas)
            : CPD(parent_node_order) {
            this->init_from_vector(lambdas);
        }

        PoissonCPD::~PoissonCPD() {}

        //----------------------------------------------------------------------
        // Copy & Move constructors/assignments
        //----------------------------------------------------------------------
        PoissonCPD::PoissonCPD(const PoissonCPD& cpd) { this->copy_cpd(cpd); }

        PoissonCPD& PoissonCPD::operator=(const PoissonCPD& cpd) {
            this->copy_cpd(cpd);
            return *this;
        }

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        void PoissonCPD::init_from_distributions(
            const vector<shared_ptr<Poisson>>& distributions) {
            this->distributions.reserve(distributions.size());
            for (const auto& distribution : distributions) {
                this->distributions.push_back(distribution);
            }
        }

        void PoissonCPD::init_from_vector(const Eigen::VectorXd& lambdas) {
            for (int i = 0; i < lambdas.rows(); i++) {
                shared_ptr<Poisson> distribution_ptr =
                    make_shared<Poisson>(lambdas(i));
                this->distributions.push_back(distribution_ptr);
            }
            this->freeze_distributions(0);
        }

        unique_ptr<CPD> PoissonCPD::clone() const {
            unique_ptr<PoissonCPD> new_cpd = make_unique<PoissonCPD>(*this);
            new_cpd->clone_distributions();
            return new_cpd;
        }

        void PoissonCPD::clone_distributions() {
            for (auto& distribution : this->distributions) {
                shared_ptr<Distribution> temp = distribution->clone();
                distribution = dynamic_pointer_cast<Poisson>(temp);
            }
        }

        string PoissonCPD::get_name() const { return "Poisson"; }

        void PoissonCPD::add_to_sufficient_statistics(
            const shared_ptr<const Distribution>& distribution,
            const vector<double>& values) {

            throw invalid_argument(
                "A Poisson distribution is not a conjugate prior of any "
                "other distribution.");
        }

        Eigen::MatrixXd PoissonCPD::sample_from_conjugacy(
            const shared_ptr<gsl_rng>& random_generator,
            int num_samples,
            const shared_ptr<const RandomVariableNode>& cpd_owner) const {

            throw invalid_argument(
                "A Poisson distribution is not a conjugate prior of any "
                "other distribution.");
        }

        void PoissonCPD::reset_sufficient_statistics() {
            // Nothing to reset
        }

        bool PoissonCPD::is_continuous() const { return false; }

    } // namespace model
} // namespace tomcat
