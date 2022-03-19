#include "Measure.h"

#include <set>

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        Measure::Measure(const shared_ptr<PGMEstimator>& estimator,
                         double threshold,
                         Estimator::FREQUENCY_TYPE frequency_type)
            : estimator(estimator), threshold(threshold),
              frequency_type(frequency_type) {

            if (estimator->is_binary_on_prediction() &&
                estimator->get_inference_horizon() > 0 &&
                estimator->get_estimates().assignment.size() == 0) {
                stringstream ss;
                ss << "It's only possible to evaluate prediction estimates for "
                      "a given assignment. Please, provide an assignment to "
                   << estimator->get_estimates().label;
                throw TomcatModelException(ss.str());
            }
        }

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        void Measure::copy_measure(const Measure& measure) {
            this->estimator = measure.estimator;
            this->threshold = measure.threshold;
        }

        vector<Eigen::MatrixXi>
        Measure::get_confusion_matrices(const EvidenceSet& test_data) const {
            const auto& true_values =
                test_data[this->estimator->get_estimates().label](0, 0);
            const auto& probabilities_per_class =
                this->estimator->get_estimates().estimates;
            int rows = (int)probabilities_per_class[0].rows();
            int cols = (int)probabilities_per_class[0].cols();
            int horizon = this->estimator->get_inference_horizon();
            int num_classes = (int)probabilities_per_class.size();
            int num_matrices = this->frequency_type == Estimator::fixed
                                   ? this->fixed_steps.size()
                                   : 1;

            vector<Eigen::MatrixXi> confusion_matrices;
            if (this->estimator->get_estimates().assignment.size() > 0) {
                confusion_matrices = vector<Eigen::MatrixXi>(
                    num_matrices, Eigen::MatrixXi::Zero(2, 2));
            }
            else {
                confusion_matrices = vector<Eigen::MatrixXi>(
                    num_matrices,
                    Eigen::MatrixXi::Zero(num_classes, num_classes));
            }

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
                        if (probabilities_per_class[0](i, valid_cols[0]) !=
                            NO_OBS) {
                            break;
                        }
                    }
                    valid_cols[0] -= horizon;
                }
                else if (this->frequency_type == Estimator::all) {
                    valid_cols =
                        vector<int>(test_data.get_num_events_for(i) - horizon);
                    for (int j = 0;
                         j < test_data.get_num_events_for(i) - horizon;
                         j++) {
                        valid_cols[j] = j;
                    }
                }
                else {
                    // Only evaluates time steps for which there are estimates
                    // for it.
                    for (int j = 0; j < cols; j++) {
                        for (const auto& estimates_per_class :
                             this->estimator->get_estimates().estimates) {
                            if (estimates_per_class(i, j) != NO_OBS) {
                                valid_cols.push_back(j);
                                break;
                            }
                        }
                    }
                }

                int fixed_time_step_idx = 0;
                for (int j : valid_cols) {
                    if (probabilities_per_class[0](i, j) == NO_OBS ||
                        (this->estimator->is_binary_on_prediction() &&
                         true_values(i, j + horizon) == NO_OBS))
                        continue;

                    if (this->estimator->get_estimates().assignment.size() >
                        0) {
                        // When an assignment is given, estimates were computed
                        // for a specific class. This is mandatory when we want
                        // to analyse predictions within a window in the future.
                        // We want to see the probability that a specific class
                        // occurred at least once in that window. The
                        // probabilities among classes, therefore, do not sum up
                        // to one.
                        int fixed_class =
                            (int)this->estimator->get_estimates().assignment(0,
                                                                             0);
                        int is_true;
                        if (horizon > 0) {
                            // Check within a window in the future instead
                            for (int t = j + 1; t <= j + horizon; t++) {
                                is_true = true_values(i, t) == fixed_class;
                                if (is_true)
                                    break;
                            }
                        }
                        else {
                            is_true = true_values(i, j) == fixed_class;
                        }
                        int is_estimation_true =
                            probabilities_per_class[0](i, j) >= this->threshold;

                        confusion_matrices[fixed_time_step_idx](
                            is_estimation_true, is_true) += 1;
                    }
                    else {
                        // When an assignment is not provided. We want to
                        // evaluate inferences (not predictions) and the sum of
                        // probabilities among classes sum up to one.
                        int estimated_class = 0;
                        double max_prob = -1;
                        for (int k = 0; k < num_classes; k++) {
                            double prob = probabilities_per_class[k](i, j);
                            if (prob > max_prob) {
                                max_prob = prob;
                                estimated_class = k;
                            }
                        }

                        if (num_classes == 2 && max_prob < this->threshold) {
                            estimated_class = 1 - estimated_class;
                        }

                        int true_class = (int)true_values(i, j);
                        if (true_class != NO_OBS) {
                            confusion_matrices[fixed_time_step_idx](
                                estimated_class, true_class) += 1;
                        }
                    }

                    if (this->frequency_type == Estimator::fixed) {
                        fixed_time_step_idx++;
                    }
                }
            }

            // Add one more matrix with the sum from each individual matrix
            if (confusion_matrices.size() > 1) {
                Eigen::MatrixXi total_confusion_matrix = Eigen::MatrixXi::Zero(
                    confusion_matrices[0].rows(), confusion_matrices[0].cols());

                for (const auto& confusion_matrix : confusion_matrices) {
                    total_confusion_matrix.array() += confusion_matrix.array();
                }

                confusion_matrices.push_back(total_confusion_matrix);
            }

            return confusion_matrices;
        }

        //----------------------------------------------------------------------
        // Getters & Setters
        //----------------------------------------------------------------------

        void
        Measure::set_fixed_steps(const unordered_set<int>& new_fixed_steps) {
            Measure::fixed_steps = new_fixed_steps;
        }

    } // namespace model
} // namespace tomcat
