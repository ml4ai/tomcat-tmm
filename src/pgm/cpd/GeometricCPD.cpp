#include "pgm/cpd/GeometricCPD.h"

#include "pgm/NumericNode.h"
#include "pgm/RandomVariableNode.h"

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        GeometricCPD::GeometricCPD(
            const vector<shared_ptr<NodeMetadata>>& parent_node_order,
            const vector<shared_ptr<Geometric>>& distributions)
            : CPD(parent_node_order) {

            this->init_from_distributions(distributions);
        }

        GeometricCPD::GeometricCPD(
            vector<shared_ptr<NodeMetadata>>&& parent_node_order,
            const vector<shared_ptr<Geometric>>& distributions)
            : CPD(parent_node_order) {

            this->init_from_distributions(distributions);
        }

        GeometricCPD::GeometricCPD(
            const vector<shared_ptr<NodeMetadata>>& parent_node_order,
            const Eigen::VectorXd& ps)
            : CPD(parent_node_order) {
            this->init_from_vector(ps);
        }

        GeometricCPD::GeometricCPD(
            vector<shared_ptr<NodeMetadata>>&& parent_node_order,
            const Eigen::VectorXd& ps)
            : CPD(parent_node_order) {
            this->init_from_vector(ps);
        }

        GeometricCPD::~GeometricCPD() {}

        //----------------------------------------------------------------------
        // Copy & Move constructors/assignments
        //----------------------------------------------------------------------
        GeometricCPD::GeometricCPD(const GeometricCPD& cpd) {
            this->copy_cpd(cpd);
        }

        GeometricCPD& GeometricCPD::operator=(const GeometricCPD& cpd) {
            this->copy_cpd(cpd);
            return *this;
        }

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        void GeometricCPD::init_from_distributions(
            const vector<shared_ptr<Geometric>>& distributions) {
            this->distributions.reserve(distributions.size());
            for (const auto& distribution : distributions) {
                this->distributions.push_back(distribution);
            }
        }

        void GeometricCPD::init_from_vector(const Eigen::VectorXd& ps) {
            for (int i = 0; i < ps.rows(); i++) {
                shared_ptr<Geometric> distribution_ptr =
                    make_shared<Geometric>(ps(i));
                this->distributions.push_back(distribution_ptr);
            }
            this->freeze_distributions(0);
        }

        unique_ptr<CPD> GeometricCPD::clone() const {
            unique_ptr<GeometricCPD> new_cpd = make_unique<GeometricCPD>(*this);
            new_cpd->clone_distributions();
            return new_cpd;
        }

        void GeometricCPD::clone_distributions() {
            for (auto& distribution : this->distributions) {
                shared_ptr<Distribution> temp = distribution->clone();
                distribution = dynamic_pointer_cast<Geometric>(temp);
            }
        }

        string GeometricCPD::get_name() const { return "Geometric"; }

        void GeometricCPD::add_to_sufficient_statistics(
            const shared_ptr<const Distribution>& distribution,
            const vector<double>& values) {

            throw invalid_argument(
                "A geometric distribution is not a conjugate prior of any "
                "other distribution.");
        }

        Eigen::MatrixXd GeometricCPD::sample_from_conjugacy(
            const shared_ptr<gsl_rng>& random_generator,
            int num_samples,
            const shared_ptr<const RandomVariableNode>& cpd_owner) const {

            throw invalid_argument(
                "A geometric distribution is not a conjugate prior of any "
                "other distribution.");
        }

        void GeometricCPD::reset_sufficient_statistics() {
            // Nothing to reset
        }

        bool GeometricCPD::is_continuous() const { return false; }

    } // namespace model
} // namespace tomcat
