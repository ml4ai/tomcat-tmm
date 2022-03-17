#include "pgm/cpd/HistogramCPD.h"

#include "pgm/NumericNode.h"
#include "pgm/RandomVariableNode.h"

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        HistogramCPD::HistogramCPD(
            const vector<shared_ptr<NodeMetadata>>& parent_node_order,
            const vector<shared_ptr<Histogram>>& distributions)
            : CPD(parent_node_order) {

            this->distributions.reserve(distributions.size());
            for (const auto& distribution : distributions) {
                this->distributions.push_back(distribution);
            }
        }

        HistogramCPD::HistogramCPD(
            vector<shared_ptr<NodeMetadata>>&& parent_node_order,
            vector<shared_ptr<Histogram>>&& distributions)
            : CPD(parent_node_order) {

            this->distributions.reserve(distributions.size());
            for (const auto& distribution : distributions) {
                this->distributions.push_back(distribution);
            }
        }

        HistogramCPD::HistogramCPD(
            const vector<shared_ptr<NodeMetadata>>& parent_node_order,
            const vector<Eigen::VectorXd>& samples_table)
            : CPD(parent_node_order) {
            this->init_from_table(samples_table);
        }

        HistogramCPD::HistogramCPD(
            vector<shared_ptr<NodeMetadata>>&& parent_node_order,
            const vector<Eigen::VectorXd>& samples_table)
            : CPD(parent_node_order) {
            this->init_from_table(samples_table);
        }

        HistogramCPD::~HistogramCPD() {}

        //----------------------------------------------------------------------
        // Copy & Move constructors/assignments
        //----------------------------------------------------------------------
        HistogramCPD::HistogramCPD(const HistogramCPD& cpd) {
            this->copy_cpd(cpd);
        }

        HistogramCPD& HistogramCPD::operator=(const HistogramCPD& cpd) {
            this->copy_cpd(cpd);
            return *this;
        }

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        void HistogramCPD::init_from_table(
            const vector<Eigen::VectorXd>& samples_table) {
            for (int row = 0; row < samples_table.size(); row++) {
                shared_ptr<Histogram> distribution_ptr =
                    make_shared<Histogram>(samples_table[row]);
                this->distributions.push_back(distribution_ptr);
            }
        }

        unique_ptr<CPD> HistogramCPD::clone() const {
            unique_ptr<HistogramCPD> new_cpd = make_unique<HistogramCPD>(*this);
            new_cpd->clone_distributions();
            return new_cpd;
        }

        void HistogramCPD::clone_distributions() {
            for (auto& distribution : this->distributions) {
                shared_ptr<Distribution> temp = distribution->clone();
                distribution = dynamic_pointer_cast<Histogram>(temp);
            }
        }

        string HistogramCPD::get_name() const { return "Histogram"; }

        void HistogramCPD::add_to_sufficient_statistics(
            const std::shared_ptr<const Distribution>& distribution,
            const vector<double>& values) {

            throw TomcatModelException("Not implemented for Histogram CPDs.");
        }

        Eigen::MatrixXd HistogramCPD::sample_from_conjugacy(
            const shared_ptr<gsl_rng>& random_generator,
            int num_samples,
            const shared_ptr<const RandomVariableNode>& cpd_owner) const {

            throw TomcatModelException("Not implemented for Histogram CPDs.");
        }

        void HistogramCPD::reset_sufficient_statistics() {
            throw TomcatModelException("Not implemented for Histogram CPDs.");
        }

        bool HistogramCPD::is_continuous() const { return true; }

        void HistogramCPD::update_sufficient_statistics(
            const shared_ptr<RandomVariableNode>& cpd_owner) {

            throw TomcatModelException("Not implemented for Histogram CPDs.");
        }

    } // namespace model
} // namespace tomcat
