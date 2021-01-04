#include "pgm/cpd/GammaCPD.h"
#include "pgm/ConstantNode.h"

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

        string GammaCPD::get_description() const {
            stringstream ss;

            ss << "Gamma CPD: {\n";
            for (auto& parameters : this->distributions) {
                ss << *parameters << "\n";
            }
            ss << "}";

            return ss.str();
        }

        void
        GammaCPD::add_to_sufficient_statistics(const vector<double>& values) {

            scoped_lock lock(*this->sufficient_statistics_mutex);
            unsigned int sum = 0;
            for (int value : values) {
                sum += value;
            }

            this->sufficient_statistics(0) += sum;
            this->sufficient_statistics(1) += values.size();
        }

        Eigen::MatrixXd GammaCPD::sample_from_conjugacy(
            const shared_ptr<gsl_rng>& random_generator,
            const vector<shared_ptr<Node>>& parent_nodes,
            int num_samples) const {

            vector<int> distribution_indices =
                this->get_indexed_distribution_indices(parent_nodes,
                                                       num_samples);

            int sample_size = this->distributions[0]->get_sample_size();

            Eigen::MatrixXd samples(distribution_indices.size(), sample_size);
            int i = 0;
            for (const auto& distribution_idx : distribution_indices) {
                Eigen::VectorXd assignment =
                    this->distributions[distribution_idx]
                        ->sample_from_conjugacy(random_generator,
                                                distribution_idx,
                                                this->sufficient_statistics);
                samples.row(i) = move(assignment);
                i++;
            }

            return samples;
        }

        void GammaCPD::reset_sufficient_statistics() {
            this->sufficient_statistics =
                Eigen::VectorXd::Zero(this->sufficient_statistics.size());
        }

    } // namespace model
} // namespace tomcat
