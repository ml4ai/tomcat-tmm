#include "EstimationProcess.h"

#include "utils/EigenExtensions.h"

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        EstimationProcess::EstimationProcess() {}

        EstimationProcess::~EstimationProcess() {}

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        void EstimationProcess::set_training_data(
            const tomcat::model::EvidenceSet& training_data) {
            for (auto& estimator : this->estimators) {
                estimator->set_training_data(training_data);
            }
        }

        void EstimationProcess::add_estimator(
            const shared_ptr<Estimator>& estimator) {
            this->estimators.push_back(estimator);
        }

        void EstimationProcess::keep_estimates() {
            for (auto& estimator : this->estimators) {
                estimator->keep_estimates();
            }
        }

        void EstimationProcess::clear_estimates() {
            for (auto& estimator : this->estimators) {
                estimator->cleanup();
            }
        }

        void EstimationProcess::prepare() {
            for (auto& estimator : this->estimators) {
                estimator->prepare();
            }
        }

        void EstimationProcess::copy_estimation(
            const EstimationProcess& estimation) {
            this->estimators = estimation.estimators;
        }

        void EstimationProcess::estimate(const shared_ptr<Estimator>& estimator,
                                         const EvidenceSet& test_data) {
            EvidenceSet new_test_data = test_data;

            for (const auto& node_label : test_data.get_node_labels()) {
                if (estimator->get_model()->has_node_with_label(node_label)) {
                    const auto& metadata =
                        estimator->get_model()->get_metadata_of(node_label);
                    if (!metadata->is_replicable() &&
                        estimator->is_computing_estimates_for(node_label)) {
                        // Remove data from nodes that only have one instance in
                        // the unrolled DBN, because the true data don't change
                        // over time and providing any data for these nodes,
                        // will make the estimator "cheat" as it will have
                        // access to the true value. For nodes that have
                        // multiple copies this is not a problem as the
                        // estimator only look at past data and, the value of
                        // that node can change over time.
                        new_test_data.remove(node_label);
                    }
                }
            }
            estimator->estimate(new_test_data);
        }

        void EstimationProcess::get_info(nlohmann::json& json) const {
            if (this->display_estimates) {
                json["estimators"] = nlohmann::json::array();
                for (const auto& estimator : this->estimators) {
                    // An estimator can be compound and comprised of multiple
                    // base estimators, which are the ones that actually
                    // store the estimates.
                    for (const auto& base_estimator :
                         estimator->get_base_estimators()) {

                        nlohmann::json json_estimator;
                        base_estimator->get_info(json_estimator);

                        CumulativeNodeEstimates cumulative_estimates =
                            base_estimator->get_cumulative_estimates();

                        json_estimator["node_label"] =
                            cumulative_estimates.label;
                        json_estimator["node_assignment"] =
                            to_string(cumulative_estimates.assignment);
                        json_estimator["executions"] = nlohmann::json::array();

                        for (const auto& estimates_matrix_per_execution :
                             cumulative_estimates.estimates) {

                            nlohmann::json json_execution;
                            json_execution["estimates"] =
                                nlohmann::json::array();

                            for (const auto& estimates_matrix :
                                 estimates_matrix_per_execution) {
                                json_execution["estimates"].push_back(
                                    to_string(estimates_matrix));
                            }

                            json_estimator["executions"].push_back(
                                json_execution);
                        }

                        json["estimators"].push_back(json_estimator);
                    }
                }
            }
        }

        //----------------------------------------------------------------------
        // Getters & Setters
        //----------------------------------------------------------------------

        void EstimationProcess::set_display_estimates(bool display_estimates) {
            EstimationProcess::display_estimates = display_estimates;
        }

        const vector<std::shared_ptr<Estimator>>&
        EstimationProcess::get_estimators() const {
            return estimators;
        }

    } // namespace model
} // namespace tomcat
