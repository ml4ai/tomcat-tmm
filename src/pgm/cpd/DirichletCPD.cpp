#include "pgm/cpd/DirichletCPD.h"

#include "pgm/ConstantNode.h"
#include "pgm/RandomVariableNode.h"

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        DirichletCPD::DirichletCPD(
            const vector<shared_ptr<NodeMetadata>>& parent_node_order,
            const vector<shared_ptr<Dirichlet>>& distributions)
            : CPD(parent_node_order) {

            this->init_from_distributions(distributions);
        }

        DirichletCPD::DirichletCPD(
            vector<shared_ptr<NodeMetadata>>&& parent_node_order,
            const vector<shared_ptr<Dirichlet>>& distributions)
            : CPD(parent_node_order) {

            this->init_from_distributions(distributions);
        }

        DirichletCPD::DirichletCPD(
            const vector<shared_ptr<NodeMetadata>>& parent_node_order,
            const Eigen::MatrixXd& alphas)
            : CPD(parent_node_order) {
            this->init_from_matrix(alphas);
        }

        DirichletCPD::DirichletCPD(
            vector<shared_ptr<NodeMetadata>>&& parent_node_order,
            const Eigen::MatrixXd& alphas)
            : CPD(parent_node_order) {
            this->init_from_matrix(alphas);
        }

        DirichletCPD::~DirichletCPD() {}

        //----------------------------------------------------------------------
        // Copy & Move constructors/assignments
        //----------------------------------------------------------------------
        DirichletCPD::DirichletCPD(const DirichletCPD& cpd) {
            this->copy_cpd(cpd);
            this->sufficient_statistics = cpd.sufficient_statistics;
        }

        DirichletCPD& DirichletCPD::operator=(const DirichletCPD& cpd) {
            this->copy_cpd(cpd);
            this->sufficient_statistics = cpd.sufficient_statistics;
            return *this;
        }

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        void DirichletCPD::init_from_distributions(
            const vector<shared_ptr<Dirichlet>>& distributions) {
            this->distributions.reserve(distributions.size());
            int sample_size = 0;
            for (const auto& distribution : distributions) {
                this->distributions.push_back(distribution);
                sample_size = distribution->get_sample_size();
            }
            this->sufficient_statistics = Eigen::VectorXd::Zero(sample_size);
        }

        void DirichletCPD::init_from_matrix(const Eigen::MatrixXd& matrix) {
            for (int i = 0; i < matrix.rows(); i++) {
                shared_ptr<Dirichlet> distribution_ptr =
                    make_shared<Dirichlet>(Dirichlet(matrix.row(i)));
                this->distributions.push_back(distribution_ptr);
            }
            this->sufficient_statistics = Eigen::VectorXd::Zero(matrix.cols());
        }

        unique_ptr<CPD> DirichletCPD::clone() const {
            unique_ptr<DirichletCPD> new_cpd = make_unique<DirichletCPD>(*this);
            new_cpd->clone_distributions();
            new_cpd->sufficient_statistics = this->sufficient_statistics;
            return new_cpd;
        }

        void DirichletCPD::clone_distributions() {
            for (auto& distribution : this->distributions) {
                shared_ptr<Distribution> temp = distribution->clone();
                distribution = dynamic_pointer_cast<Dirichlet>(temp);
            }
        }

        string DirichletCPD::get_name() const {
            return "Dirichlet";
        }

        void DirichletCPD::add_to_sufficient_statistics(
            const vector<double>& values) {

            scoped_lock lock(*this->sufficient_statistics_mutex);
            for (int value : values) {
                this->sufficient_statistics(value) += 1;
            }
        }

        Eigen::MatrixXd DirichletCPD::sample_from_conjugacy(
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
                Eigen::VectorXd assignment =
                    this->distributions[distribution_idx]
                        ->sample_from_conjugacy(random_generator,
                                                distribution_idx,
                                                this->sufficient_statistics);
                samples.row(i) = move(assignment);
            }

            return samples;
        }

        void DirichletCPD::reset_sufficient_statistics() {
            this->sufficient_statistics =
                Eigen::VectorXd::Zero(this->sufficient_statistics.size());
        }

        bool DirichletCPD::is_continuous() const {
            return false;
        }

    } // namespace model
} // namespace tomcat
