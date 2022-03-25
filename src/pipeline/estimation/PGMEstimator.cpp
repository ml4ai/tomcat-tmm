#include "PGMEstimator.h"

#include <sstream>

#include "utils/EigenExtensions.h"

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        PGMEstimator::PGMEstimator(const shared_ptr<DynamicBayesNet>& model,
                                   int inference_horizon,
                                   const std::string& node_label,
                                   const Eigen::VectorXd& assignment)
            : Estimator(model), inference_horizon(inference_horizon),
              compound(false) {

            const auto& metadata = model->get_metadata_of(node_label);
            if (!metadata->is_replicable() && inference_horizon > 0) {
                throw TomcatModelException("Inference horizon for "
                                           "non-replicable nodes can only be "
                                           "0.");
            }

            if (assignment.size() == 0 && metadata->get_cardinality() <= 1) {
                throw TomcatModelException(
                    "An assignment has to be provided for nodes with "
                    "continuous distributions.");
            }

            this->estimates.label = node_label;
            this->estimates.assignment = assignment;
            this->cumulative_estimates.label = node_label;
            this->cumulative_estimates.assignment = assignment;
        }

        PGMEstimator::PGMEstimator(const shared_ptr<DynamicBayesNet>& model)
            : Estimator(model), compound(true) {}

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        void PGMEstimator::copy(const PGMEstimator& estimator) {
            Estimator::copy(estimator);
            this->estimates = estimator.estimates;
            this->cumulative_estimates = estimator.cumulative_estimates;
            this->inference_horizon = estimator.inference_horizon;
        }

        NodeEstimates PGMEstimator::get_estimates_at(int time_step) const {
            if (this->estimates.estimates[0].cols() <= time_step) {
                stringstream ss;
                ss << "The chosen PGMEstimator can only calculate estimates "
                      "up to time step "
                   << time_step;
                throw out_of_range(ss.str());
            }

            NodeEstimates sliced_estimates;
            sliced_estimates.label = this->estimates.label;
            sliced_estimates.assignment = this->estimates.assignment;
            for (const auto& estimates_per_assignment :
                 this->estimates.estimates) {
                sliced_estimates.estimates.push_back(
                    estimates_per_assignment.col(time_step));
            }

            return sliced_estimates;
        }

        void PGMEstimator::prepare() {
            // Clear estimates so they can be recalculated over the new
            // training data in the next call to the function estimate.
            this->estimates.estimates.clear();
            this->estimates.custom_data.clear();

            int k = 1;
            if (this->estimates.assignment.size() == 0) {
                k = dynamic_pointer_cast<DynamicBayesNet>(this->model)
                        ->get_cardinality_of(this->estimates.label);
            }
            this->estimates.estimates.resize(k);
        }

        void PGMEstimator::keep_estimates() {
            this->cumulative_estimates.estimates.push_back({});
            int i = (int)this->cumulative_estimates.estimates.size() - 1;
            for (const auto& estimate : this->estimates.estimates) {
                this->cumulative_estimates.estimates[i].push_back(estimate);
            }
            this->cumulative_estimates.custom_data.push_back({});
            for (const auto& custom_data : this->estimates.custom_data) {
                this->cumulative_estimates.custom_data[i].push_back(
                    custom_data);
            }
        }

        void PGMEstimator::get_info(nlohmann::json& json_estimators) const {
            nlohmann::json json_estimator;
            CumulativeNodeEstimates cumulative_estimates =
                this->get_cumulative_estimates();

            json_estimator["name"] = this->get_name();
            json_estimator["inference_horizon"] = this->inference_horizon;
            json_estimator["node_label"] = cumulative_estimates.label;
            json_estimator["node_assignment"] =
                to_string(cumulative_estimates.assignment);
            json_estimator["executions"] = nlohmann::json::array();

            // Estimates
            for (const auto& estimates_matrix_per_execution :
                 cumulative_estimates.estimates) {

                nlohmann::json json_execution;
                json_execution["estimates"] = nlohmann::json::array();

                for (const auto& estimates_matrix :
                     estimates_matrix_per_execution) {
                    json_execution["estimates"].push_back(
                        to_string(estimates_matrix));
                }

                json_estimator["executions"].push_back(json_execution);
            }

            // Custom data
            for (const auto& custom_data_matrix_per_execution :
                 cumulative_estimates.custom_data) {

                nlohmann::json json_execution;
                json_execution["custom_data"] = nlohmann::json::array();

                for (const auto& custom_data_matrix :
                     custom_data_matrix_per_execution) {
                    json_execution["custom_data"].push_back(
                        to_string(custom_data_matrix));
                }

                json_estimator["executions"].push_back(json_execution);
            }

            json_estimators.push_back(json_estimator);
        }

        void PGMEstimator::cleanup() {
            this->estimates.estimates.clear();
            this->cumulative_estimates.estimates.clear();
            this->cumulative_estimates.custom_data.clear();
        }

        bool PGMEstimator::is_computing_estimates_for(
            const std::string& node_label) const {
            return this->estimates.label == node_label;
        }

        bool PGMEstimator::is_binary_on_prediction() const { return true; }

        //----------------------------------------------------------------------
        // Getters & Setters
        //----------------------------------------------------------------------
        NodeEstimates PGMEstimator::get_estimates() const {
            return this->estimates;
        }

        CumulativeNodeEstimates PGMEstimator::get_cumulative_estimates() const {
            return this->cumulative_estimates;
        }

        int PGMEstimator::get_inference_horizon() const {
            return inference_horizon;
        }

        bool PGMEstimator::is_compound() const { return compound; }

        vector<shared_ptr<PGMEstimator>> PGMEstimator::get_base_estimators() {
            vector<shared_ptr<PGMEstimator>> base_estimators;
            base_estimators.push_back(shared_from_this());
            return base_estimators;
        }
    } // namespace model
} // namespace tomcat
