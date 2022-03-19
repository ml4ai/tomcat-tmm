#include "RMSE.h"

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        RMSE::RMSE(const shared_ptr<PGMEstimator>& estimator,
                   Estimator::FREQUENCY_TYPE frequency_type)
            : Measure(estimator, 0, frequency_type) {}

        RMSE::~RMSE() {}

        //----------------------------------------------------------------------
        // Copy & Move constructors/assignments
        //----------------------------------------------------------------------
        RMSE::RMSE(const RMSE& rmse) { this->copy_measure(rmse); }

        RMSE& RMSE::operator=(const RMSE& rmse) {
            this->copy_measure(rmse);
            return *this;
        }

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        NodeEvaluation RMSE::evaluate(const EvidenceSet& test_data) const {
            vector<NodeEvaluation> evaluations;

            NodeEstimates estimates = this->estimator->get_estimates();

            NodeEvaluation evaluation;
            evaluation.label = estimates.label;
            evaluation.assignment = estimates.assignment;
            evaluation.evaluation = Eigen::MatrixXd::Constant(1, 1, NO_OBS);

            if (test_data.has_data_for(evaluation.label)) {
                int rows = estimates.estimates[0].rows();
                int cols = estimates.estimates[0].cols();
                int horizon = this->estimator->get_inference_horizon();
                int num_measures = this->frequency_type == Estimator::fixed
                                       ? this->fixed_steps.size()
                                       : 1;

                // Per fixed time
                vector<double> square_diffs(num_measures, 0);
                vector<double> num_cases(num_measures, 0);
                for (int i = 0; i < rows; i++) {
                    // Convert time step to column index
                    vector<int> valid_cols;
                    if (this->frequency_type == Estimator::fixed) {
                        for (int t : this->fixed_steps) {
                            valid_cols.push_back(
                                test_data.get_column_index_for(i, t));
                        }
                        sort(valid_cols.begin(), valid_cols.end());
                    }
                    else if (this->frequency_type == Estimator::last) {
                        valid_cols = vector<int>(1);
                        for (valid_cols[0] = cols - 1; valid_cols[0] >= 0;
                             valid_cols[0]--) {
                            if (estimates.estimates[0](i, valid_cols[0]) !=
                                NO_OBS) {
                                break;
                            }
                        }
                        valid_cols[0] -= horizon;
                    }
                    else {
                        valid_cols = vector<int>(cols);
                        for (int j = 0;
                             j < test_data.get_num_events_for(i) - horizon;
                             j++) {
                            valid_cols[j] = j;
                        }
                    }

                    int fixed_time_step_idx = 0;
                    for (int j : valid_cols) {
                        if (estimates.estimates[0](i, j) == NO_OBS ||
                            test_data[evaluation.label](0, 0)(i, j + horizon) ==
                                NO_OBS)
                            continue;

                        square_diffs[fixed_time_step_idx] +=
                            pow(estimates.estimates[0](i, j) -
                                    test_data[evaluation.label](0, 0)(i, j),
                                2);
                        num_cases[fixed_time_step_idx] += 1;

                        if (this->frequency_type == Estimator::fixed) {
                            fixed_time_step_idx++;
                        }
                    }
                }

                if (num_measures > 1) {
                    // We compute RMSE for each fixed time step and add
                    // one more with the total
                    evaluation.evaluation =
                        Eigen::MatrixXd::Constant(1, num_measures + 1, NO_OBS);
                }
                else {
                    evaluation.evaluation =
                        Eigen::MatrixXd::Constant(1, num_measures, NO_OBS);
                }

                double total_square_diff = 0;
                double total_cases = 0;
                for (int i = 0; i < num_measures; i++) {
                    double rmse = sqrt(square_diffs[i] / num_cases[i]);
                    evaluation.evaluation(0, i) = rmse;

                    total_square_diff += square_diffs[i];
                    total_cases += num_cases[i];
                }

                if (num_measures > 1) {
                    double total_rmse = sqrt(total_square_diff / total_cases);
                    evaluation.evaluation(0, num_measures) = total_rmse;
                }
            }

            return evaluation;
        }

        void RMSE::get_info(nlohmann::json& json) const {
            json["name"] = NAME;
            json["threshold"] = this->threshold;
            this->estimator->get_info(json["estimator"]);
        }

    } // namespace model
} // namespace tomcat
