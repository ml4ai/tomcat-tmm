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
        // TODO - update sEstimator to receive a range
        SamplerEstimator::SamplerEstimator(
            const shared_ptr<DynamicBayesNet>& model,
            int inference_horizon,
            const std::string& node_label,
            const Eigen::VectorXd& low,
            const Eigen::VectorXd& high)
            : Estimator(model, inference_horizon, node_label, low),
              update_estimates_mutex(make_unique<mutex>()) {

            if (node_label == "FinalScore") {
                this->custom_metric = make_shared<FinalScore>();
            }

            if (low.size() == 0 && !this->custom_metric) {
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
            // TODO - remove this function if there's nothing to copy
        }

        void SamplerEstimator::prepare() {
            Estimator::prepare();
            // TODO - remove this function if there's nothing to do here.
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

        void SamplerEstimator::estimate(const std::shared_ptr<Sampler>& sampler,
                                        int data_point_idx,
                                        int time_step) {
            if (this->estimates.label == "FinalScore") {
                if (this->estimates.estimates.at(0).size() == 0) {
                    this->estimates.estimates[0] = Eigen::MatrixXd::Zero(
                        sampler->get_num_samples(),
                        sampler->get_model()->get_time_steps());
                }

                double final_score =
                    this->custom_metric->calculate(sampler, time_step).at(0);
                this->estimates.estimates.at(0)(data_point_idx, time_step) =
                    final_score;
            }
            else {
                const auto& node_metadata =
                    this->model->get_metadata_of(this->estimates.label);
                int node_initial_time_step =
                    node_metadata->get_initial_time_step();
                Eigen::MatrixXd samples(0, 0);
                if (this->inference_horizon > 0) {
                    samples = sampler->get_samples(this->estimates.label)(0, 0);
                    samples = samples.block(0,
                                            time_step + 1,
                                            samples.rows(),
                                            this->inference_horizon);
                }
                else {
                    if (time_step >= node_initial_time_step) {
                        samples =
                            sampler->get_samples(this->estimates.label)(0, 0)
                                .col(node_initial_time_step);
                    }
                }

                int k = 1;
                double low;
                double high;
                if (this->estimates.assignment.size() == 0) {
                    // For each possible discrete value the node can take, we
                    // compute the probability estimate for the node.
                    k = sampler->get_model()->get_cardinality_of(
                        this->estimates.label);
                }
                else {
                    low = this->estimates.assignment(0);
                    high = this->estimates.assignment(0);
                }

                if (this->estimates.estimates.empty()) {
                    this->estimates.estimates.resize(k);
                }

                for (int i = 0; i < k; i++) {
                    double probability;
                    if (samples.size() == 0) {
                        probability = NO_OBS;
                    }
                    else {
                        if (k > 1) {
                            low = i;
                            high = i;
                        }

                        probability =
                            this->get_probability_in_range(samples, low, high);
                    }

                    // Add probability to the estimates
                    auto& estimates_matrix = this->estimates.estimates.at(i);
                    int new_rows = data_point_idx + 1;
                    int new_cols =
                        max((int)estimates_matrix.cols(), time_step + 1);

                    if (estimates_matrix.rows() != new_rows ||
                        estimates_matrix.cols() != new_cols) {
                        estimates_matrix.conservativeResize(new_rows, new_cols);
                    }

                    estimates_matrix(data_point_idx, time_step) = probability;
                }
            }
        }

        double SamplerEstimator::get_probability_in_range(
            const Eigen::MatrixXd& samples, double low, double high) const {

            Eigen::VectorXd matches;
            double probability;
            if (low == high) {
                matches =
                    ((samples.array() == low).cast<int>().rowwise().sum() > 0)
                        .cast<double>();
            }
            else {
                Eigen::VectorXi matches_low =
                    ((samples.array() >= low).cast<int>().rowwise().sum() > 0)
                        .cast<int>();

                Eigen::VectorXi matches_high =
                    ((samples.array() <= high).cast<int>().rowwise().sum() > 0)
                        .cast<int>();
                matches = (matches_low.array() == matches_high.array())
                              .cast<double>();
            }

            probability = matches.mean();
            return probability;
        }

    } // namespace model
} // namespace tomcat
