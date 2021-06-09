#include "SamplerEstimator.h"

#include <iostream>
#include <thread>

#include "pipeline/estimation/custom_metrics/FinalScore.h"
#include "utils/EigenExtensions.h"

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        SamplerEstimator::SamplerEstimator() {}

        SamplerEstimator::SamplerEstimator(
            const shared_ptr<DynamicBayesNet>& model,
            int inference_horizon,
            const std::string& node_label,
            const Eigen::VectorXd& low,
            const Eigen::VectorXd& high)
            : Estimator(model, inference_horizon, node_label, low) {

            if (low.size() == 0) {
                int cardinality = model->get_cardinality_of(node_label);
                this->estimates.estimates =
                    vector<Eigen::MatrixXd>(cardinality);
            }
            else {
                this->estimates.estimates = vector<Eigen::MatrixXd>(1);
            }
        }

        SamplerEstimator::~SamplerEstimator() {}

        //----------------------------------------------------------------------
        // Copy & Move constructors/assignments
        //----------------------------------------------------------------------
        SamplerEstimator::SamplerEstimator(const SamplerEstimator& estimator) {
            Estimator::copy_estimator(estimator);
            this->copy_estimator(estimator);
        }

        SamplerEstimator&
        SamplerEstimator::operator=(const SamplerEstimator& estimator) {
            Estimator::copy_estimator(estimator);
            this->copy_estimator(estimator);
            return *this;
        }

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        void SamplerEstimator::copy(const SamplerEstimator& estimator) {
            Estimator::copy_estimator(estimator);
        }

        void SamplerEstimator::estimate(const EvidenceSet& new_data) {
            new TomcatModelException(
                "Whenever working with sampler estimators, "
                "please use the alternative estimate "
                "function.");
        }

        void SamplerEstimator::get_info(nlohmann::json& json) const {
            json["name"] = this->get_name();
            json["inference_horizon"] = this->inference_horizon;
        }

        string SamplerEstimator::get_name() const { return "sampler"; }

        void SamplerEstimator::estimate(const EvidenceSet& particles,
                                        const EvidenceSet& projected_particles,
                                        int data_point_idx,
                                        int time_step) {

            if (this->inference_horizon == 0) {
                const Tensor3 samples_tensor = particles[this->estimates.label];

                for (int t = 0; t < particles.get_time_steps(); t++) {
                    // Each dimensionality is computed individually
                    const auto& metadata = this->get_model()->get_metadata_of(
                        this->estimates.label);

                    if (this->estimates.assignment.size() == 0) {
                        // Compute estimates for each value the node can take
                        // (assuming the node's distribution is discrete)
                        Eigen::MatrixXd samples = samples_tensor(0, 0);

                        int k = this->get_model()->get_cardinality_of(
                            this->estimates.label);
                        Eigen::VectorXd probs = Eigen::VectorXd::Zero(k);

                        const auto& metadata =
                            this->get_model()->get_metadata_of(
                                this->estimates.label);
                        if (time_step >= metadata->get_initial_time_step()) {
                            for (int i = 0; i < samples.rows(); i++) {
                                probs[samples(i, t)] += 1;
                            }

                            probs /= samples.rows();
                        }

                        for (int i = 0; i < k; i++) {
                            this->update_estimates(
                                i, data_point_idx, time_step, probs[i]);
                        }
                    }
                    else {
                        double low = this->estimates.assignment[0];
                        // TODO - change this when range is implemented
                        double high = low;

                        double prob = this->get_probability_in_range(
                            samples_tensor.col(t), low, high);
                        this->update_estimates(
                            0, data_point_idx, time_step, prob);
                    }

                    time_step++;
                }
            }
            else {
                const Tensor3 samples_tensor =
                    projected_particles[this->estimates.label];

                if (this->estimates.assignment.size() == 0) {
                    // Compute estimates for each value the node can take
                    // (assuming the node's distribution is discrete)
                    int k = this->get_model()->get_cardinality_of(
                        this->estimates.label);

                    for (int i = 0; i < k; i++) {
                        double prob = this->get_probability_in_range(
                            samples_tensor, i, i);
                        this->update_estimates(
                            i, data_point_idx, time_step, prob);
                    }
                }
                else {
                    double low = this->estimates.assignment[0];
                    // TODO - change this when range is implemented
                    double high = low;
                    double prob = this->get_probability_in_range(
                        samples_tensor, low, high);
                    this->update_estimates(0, data_point_idx, time_step, prob);
                }
            }
        }

        double SamplerEstimator::get_probability_in_range(
            const Tensor3& samples, double low, double high) const {

            Eigen::VectorXi matches =
                Eigen::VectorXi::Ones(samples.get_shape()[1]);
            double probability;
            for (int i = 0; i < samples.get_shape()[0]; i++) {
                Eigen::MatrixXd matrix = samples(i, 0);

                if (low == high) {
                    matches =
                        (matches.array() ==
                         ((matrix.array() == low).cast<int>().rowwise().sum() >
                          0)
                             .cast<int>())
                            .cast<int>();
                }
                else {
                    Eigen::VectorXi matches_low =
                        (matches.array() ==
                         ((matrix.array() >= low).cast<int>().rowwise().sum() >
                          0)
                             .cast<int>())
                            .cast<int>();

                    Eigen::VectorXi matches_high =
                        (matches.array() ==
                         ((matrix.array() <= high).cast<int>().rowwise().sum() >
                          0)
                             .cast<int>())
                            .cast<int>();
                    matches = (matches_low.array() == matches_high.array())
                                  .cast<int>();
                }
            }

            probability = matches.cast<double>().mean();
            return probability;
        }

        void SamplerEstimator::update_estimates(int estimates_idx,
                                                int data_point_idx,
                                                int time_step,
                                                double probability) {
            auto& estimates_matrix =
                this->estimates.estimates.at(estimates_idx);
            int new_rows =
                max(data_point_idx + 1, (int)estimates_matrix.rows());
            int new_cols = max(time_step + 1, (int)estimates_matrix.cols());
            estimates_matrix.conservativeResize(new_rows, new_cols);
            estimates_matrix(data_point_idx, time_step) = probability;
        }

    } // namespace model
} // namespace tomcat
