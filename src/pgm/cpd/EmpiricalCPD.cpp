#include "pgm/cpd/EmpiricalCPD.h"

#include "pgm/NumericNode.h"
#include "pgm/RandomVariableNode.h"

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        EmpiricalCPD::EmpiricalCPD(
            const vector<shared_ptr<NodeMetadata>>& parent_node_order,
            const vector<shared_ptr<Empirical>>& distributions)
            : CPD(parent_node_order) {

            this->distributions.reserve(distributions.size());
            for (const auto& distribution : distributions) {
                this->distributions.push_back(distribution);
            }
        }

        EmpiricalCPD::EmpiricalCPD(
            vector<shared_ptr<NodeMetadata>>&& parent_node_order,
            vector<shared_ptr<Empirical>>&& distributions)
            : CPD(parent_node_order) {

            this->distributions.reserve(distributions.size());
            for (const auto& distribution : distributions) {
                this->distributions.push_back(distribution);
            }
        }

        EmpiricalCPD::EmpiricalCPD(
            const vector<shared_ptr<NodeMetadata>>& parent_node_order,
            const vector<Eigen::VectorXd>& samples_table)
            : CPD(parent_node_order) {
            this->init_from_table(samples_table);
        }

        EmpiricalCPD::EmpiricalCPD(
            vector<shared_ptr<NodeMetadata>>&& parent_node_order,
            const vector<Eigen::VectorXd>& samples_table)
            : CPD(parent_node_order) {
            this->init_from_table(samples_table);
        }

        EmpiricalCPD::~EmpiricalCPD() {}

        //----------------------------------------------------------------------
        // Copy & Move constructors/assignments
        //----------------------------------------------------------------------
        EmpiricalCPD::EmpiricalCPD(const EmpiricalCPD& cpd) {
            this->copy_cpd(cpd);
        }

        EmpiricalCPD& EmpiricalCPD::operator=(const EmpiricalCPD& cpd) {
            this->copy_cpd(cpd);
            return *this;
        }

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        void EmpiricalCPD::init_from_table(
            const vector<Eigen::VectorXd>& samples_table) {
            for (int row = 0; row < samples_table.size(); row++) {
                shared_ptr<Empirical> distribution_ptr =
                    make_shared<Empirical>(samples_table[row]);
                this->distributions.push_back(distribution_ptr);
            }
        }

        unique_ptr<CPD> EmpiricalCPD::clone() const {
            unique_ptr<EmpiricalCPD> new_cpd = make_unique<EmpiricalCPD>(*this);
            new_cpd->clone_distributions();
            return new_cpd;
        }

        void EmpiricalCPD::clone_distributions() {
            for (auto& distribution : this->distributions) {
                shared_ptr<Distribution> temp = distribution->clone();
                distribution = dynamic_pointer_cast<Empirical>(temp);
            }
        }

        string EmpiricalCPD::get_name() const { return "Empirical"; }

        void EmpiricalCPD::add_to_sufficient_statistics(
            const std::shared_ptr<const Distribution>& distribution,
            const vector<double>& values) {

            throw TomcatModelException("Not implemented for Empirical CPDs.");
        }

        Eigen::MatrixXd EmpiricalCPD::sample_from_conjugacy(
            const shared_ptr<gsl_rng>& random_generator,
            int num_samples,
            const shared_ptr<const RandomVariableNode>& cpd_owner) const {

            throw TomcatModelException("Not implemented for Empirical CPDs.");
        }

        void EmpiricalCPD::reset_sufficient_statistics() {
            throw TomcatModelException("Not implemented for Empirical CPDs.");
        }

        bool EmpiricalCPD::is_continuous() const { return true; }

        void EmpiricalCPD::update_sufficient_statistics(
            const shared_ptr<RandomVariableNode>& cpd_owner) {

            throw TomcatModelException("Not implemented for Empirical CPDs.");
        }

    } // namespace model
} // namespace tomcat
