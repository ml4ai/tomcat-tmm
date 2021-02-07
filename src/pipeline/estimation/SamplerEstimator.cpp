#include "SamplerEstimator.h"

#include <iostream>
#include <thread>

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

        vector<double>
        SamplerEstimator::estimate(const std::shared_ptr<Sampler>& sampler,
                                   int data_point_idx,
                                   int time_step) {

            // Slicing the matrix directly is more efficient than calling the
            // get_samples with a range here because it will trye to slice a
            // tensor, but we know that in this scenario this tensor has
            // an unitary first dimension since the nodes being estimated are
            // data nodes and not parameter ones.
            Eigen::MatrixXd samples =
                sampler->get_samples(this->estimates.label)(0, 0);
            const auto& node_metadata =
                this->model->get_metadata_of(this->estimates.label);
            int node_initial_time_step = node_metadata->get_initial_time_step();
            if (node_metadata->is_replicable()) {
                int initial_col =
                    this->inference_horizon == 0 ? time_step : time_step + 1;
                int num_cols =
                    this->inference_horizon == 0 ? 1 : this->inference_horizon;
                samples =
                    samples.block(0, initial_col, samples.rows(), num_cols);
            }
            else {
                samples =
                    samples.block(0, node_initial_time_step, samples.rows(), 1);
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

            vector<double> probabilities_per_class(k);
            for (int i = 0; i < k; i++) {
                double probability = NO_OBS;
                if (node_initial_time_step <=
                    time_step + this->inference_horizon) {
                    if (k > 1) {
                        low = i;
                        high = i;
                    }
                    probability =
                        this->get_probability_in_range(samples, low, high);
                }

                probabilities_per_class[i] = probability;
            }

            return probabilities_per_class;
        }

        void SamplerEstimator::set_estimates(
            const std::vector<Eigen::VectorXd>& probabilities_per_class,
            int initial_data_idx,
            int final_data_idx,
            int time_step) {

            int data_size = final_data_idx - initial_data_idx;

            scoped_lock lock(*this->update_estimates_mutex);
            for (int k = 0; k < probabilities_per_class.size(); k++) {
                auto& estimates_matrix = this->estimates.estimates.at(k);
                int new_rows =
                    max((int)estimates_matrix.rows(), final_data_idx);
                int new_cols = max((int)estimates_matrix.cols(), time_step + 1);

                if (estimates_matrix.rows() != new_rows ||
                    estimates_matrix.cols() != new_cols) {
                    estimates_matrix.conservativeResize(new_rows, new_cols);
                }

                estimates_matrix.block(
                    initial_data_idx, time_step, data_size, 1) =
                    probabilities_per_class.at(k);
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
