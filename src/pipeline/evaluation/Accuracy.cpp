#include "Accuracy.h"

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        Accuracy::Accuracy(const shared_ptr<Estimator>& estimator,
                           double threshold,
                           FREQUENCY_TYPE frequency_type)
            : Measure(estimator, threshold, frequency_type) {}

        Accuracy::~Accuracy() {}

        //----------------------------------------------------------------------
        // Copy & Move constructors/assignments
        //----------------------------------------------------------------------
        Accuracy::Accuracy(const Accuracy& accuracy) {
            this->copy_measure(accuracy);
        }

        Accuracy& Accuracy::operator=(const Accuracy& accuracy) {
            this->copy_measure(accuracy);
            return *this;
        }

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        NodeEvaluation Accuracy::evaluate(const EvidenceSet& test_data) const {
            vector<NodeEvaluation> evaluations;

            NodeEstimates estimates = this->estimator->get_estimates();

            NodeEvaluation evaluation;
            evaluation.label = estimates.label;
            evaluation.assignment = estimates.assignment;
            evaluation.evaluation = Eigen::MatrixXd::Constant(1, 1, NO_OBS);

            if (test_data.has_data_for(evaluation.label)) {
                // Get matrix of true observations.
                Tensor3 real_data_3d = test_data[evaluation.label];
                Eigen::MatrixXd true_values = real_data_3d(0, 0);

                int cols = estimates.estimates[0].cols();
                int first_valid_time_step =
                    EvidenceSet::get_first_time_with_observation(real_data_3d);
                vector<vector<int>> time_steps_per_point(
                    test_data.get_num_data_points());

                int num_accuracies = 1;
                if (this->frequency_type == all) {
                    if (test_data.is_event_based()) {
                        for (int d = 0; d < test_data.get_num_data_points();
                             d++) {
                            for (int t = 0; t < test_data.get_num_events_for(d);
                                 t++) {
                                time_steps_per_point[d].push_back(t);
                            }
                        }
                    }
                    else {
                        vector<int> time_steps;
                        for (int t = first_valid_time_step; t < cols; t++) {
                            time_steps.push_back(t);
                        }
                        for (int d = 0; d < test_data.get_num_data_points();
                             d++) {
                            time_steps_per_point[d] = time_steps;
                        }
                    }
                }
                else if (this->frequency_type == last) {
                    if (test_data.is_event_based()) {
                        for (int d = 0; d < test_data.get_num_data_points();
                             d++) {
                            int max_event_idx = cols - 1;
                            int last_event_idx = test_data.get_column_index_for(
                                d, max_event_idx);
                            time_steps_per_point[d].push_back(last_event_idx);
                        }
                    }
                    else {
                        vector<int> time_steps(1, cols - 1);
                        for (int d = 0; d < test_data.get_num_data_points();
                             d++) {
                            time_steps_per_point[d] = time_steps;
                        }
                    }
                }
                else {
                    num_accuracies = this->fixed_steps.size() > 1
                                         ? this->fixed_steps.size() + 1
                                         : this->fixed_steps.size();
                    if (test_data.is_event_based()) {
                        for (int d = 0; d < test_data.get_num_data_points();
                             d++) {
                            for (int t : this->fixed_steps) {
                                time_steps_per_point[d].push_back(
                                    test_data.get_column_index_for(d, t));
                            }
                        }
                    }
                    else {
                        for (int d = 0; d < test_data.get_num_data_points();
                             d++) {
                            time_steps_per_point[d] = this->fixed_steps;
                        }
                    }
                }

                if (estimates.assignment.size() == 0 ||
                    test_data.is_event_based()) {

                    Eigen::MatrixXd accuracies =
                        Eigen::MatrixXd::Zero(1, num_accuracies);
                    int total_cases = 0;

                    for (int d = 0; d < test_data.get_num_data_points(); d++) {
                        int acc = 0;
                        int last_col = -1;
                        for (int t : time_steps_per_point[d]) {
                            // Each element of the vector estimates.estimates
                            // represents a possible assignment a node can have
                            // (0, 1, ..., cardinality - 1), and it contains a
                            // matrix of estimated probability for such
                            // assignment, where the rows index a data point and
                            // the columns index a time step. We loop over the
                            // elements of the vector for a given data point and
                            // time, to compute the estimated assignment by
                            // choosing the one with highest probability among
                            // the other assignments' estimate for the same data
                            // point and time step.
                            int inferred_assignment = 0;
                            double max_prob = 0;
                            for (int i = 0; i < estimates.estimates.size();
                                 i++) {
                                double prob = estimates.estimates[i](d, t);
                                if (prob > max_prob) {
                                    max_prob = prob;
                                    inferred_assignment = i;
                                }
                            }

                            int true_assignment = true_values(d, t);
                            if (inferred_assignment == true_assignment) {
                                if (num_accuracies == 1) {
                                    accuracies(0, 0) = accuracies(0, 0) + 1;
                                }
                                else {
                                    accuracies(0, acc) = accuracies(0, acc) + 1;
                                }
                            }
                            acc++;
                            total_cases++;
                        }
                    }

                    if (this->frequency_type == all) {
                        double accuracy = accuracies(0, 0) / total_cases;
                        evaluation.evaluation =
                            Eigen::MatrixXd::Constant(1, 1, accuracy);
                    }
                    else {
                        accuracies.array() /= test_data.get_num_data_points();
                        if (num_accuracies > 1) {
                            // Add the aggregated accuracy in the last position
                            accuracies(0, num_accuracies - 1) =
                                accuracies.sum() / (num_accuracies - 1);
                        }
                        evaluation.evaluation = accuracies;
                    }
                }
                else {
                    vector<int> time_steps = time_steps_per_point[0];
                    if (frequency_type == all) {
                        ConfusionMatrix confusion_matrix =
                            this->get_confusion_matrix(estimates.estimates[0],
                                                       true_values,
                                                       estimates.assignment[0]);
                        double accuracy = this->get_score(confusion_matrix);
                        evaluation.evaluation =
                            Eigen::MatrixXd::Constant(1, 1, accuracy);
                    }
                    else {
                        Eigen::MatrixXd accuracies =
                            Eigen::MatrixXd::Zero(1, num_accuracies);

                        Eigen::MatrixXd sliced_estimates(
                            estimates.estimates[0].rows(), time_steps.size());
                        Eigen::MatrixXd sliced_true_values(
                            estimates.estimates[0].rows(), time_steps.size());

                        int i = 0;
                        for (int t : time_steps) {
                            sliced_estimates.col(i) =
                                estimates.estimates[0].col(t);
                            sliced_true_values.col(i) = true_values.col(t);

                            ConfusionMatrix confusion_matrix =
                                this->get_confusion_matrix(
                                    sliced_estimates,
                                    sliced_true_values,
                                    estimates.assignment[0]);
                            accuracies(0, i) =
                                this->get_score(confusion_matrix);
                            i++;
                        }

                        if (num_accuracies > 1) {
                            ConfusionMatrix confusion_matrix =
                                this->get_confusion_matrix(
                                    sliced_estimates,
                                    sliced_true_values,
                                    estimates.assignment[0]);
                            accuracies(0, num_accuracies - 1) =
                                this->get_score(confusion_matrix);
                        }

                        evaluation.evaluation = accuracies;
                    }
                }
            }

            return evaluation;
        }

        double
        Accuracy::get_score(const ConfusionMatrix& confusion_matrix) const {
            double accuracy = (confusion_matrix.true_positives +
                               confusion_matrix.true_negatives) /
                              (double)confusion_matrix.get_total();

            return accuracy;
        }

        void Accuracy::get_info(nlohmann::json& json) const {
            json["name"] = NAME;
            json["threshold"] = this->threshold;
            this->estimator->get_info(json["estimator"]);
        }

    } // namespace model
} // namespace tomcat
