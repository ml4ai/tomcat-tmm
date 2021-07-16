#include "F1Score.h"

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        F1Score::F1Score(const shared_ptr<Estimator>& estimator,
                         double threshold)
            : Measure(estimator, threshold, last) {

            if (estimator->get_inference_horizon() == 0 and
                estimator->get_estimates().assignment.size() == 0) {
                throw TomcatModelException(
                    "F1 Score cannot be used for multiclass inference.");
            }
        }

        F1Score::~F1Score() {}

        //----------------------------------------------------------------------
        // Copy & Move constructors/assignments
        //----------------------------------------------------------------------
        F1Score::F1Score(const F1Score& f1_score) {
            this->copy_measure(f1_score);
        }

        F1Score& F1Score::operator=(const F1Score& f1_score) {
            this->copy_measure(f1_score);
            return *this;
        }

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        NodeEvaluation F1Score::evaluate(const EvidenceSet& test_data) const {
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
                unordered_set<int> time_steps;
                if (this->frequency_type == last) {
                    time_steps.insert(cols - 1);
                }
                else {
                    time_steps = this->fixed_steps;
                }

                if (frequency_type == all) {
                    ConfusionMatrix confusion_matrix =
                        this->get_confusion_matrix(estimates.estimates[0],
                                                   true_values,
                                                   estimates.assignment[0]);
                    double f1 = this->get_score(confusion_matrix);
                    evaluation.evaluation = Eigen::MatrixXd::Constant(1, 1, f1);
                }
                else {
                    if (time_steps.empty())
                        return evaluation;

                    // If the frequency is fixed, we compute accuracies per time
                    // step in the fixed list and a final total one
                    int num_f1s =
                        this->frequency_type == fixed && time_steps.size() > 1
                        ? time_steps.size() + 1
                        : 1;
                    Eigen::MatrixXd f1s = Eigen::MatrixXd::Zero(1, num_f1s);

                    Eigen::MatrixXd sliced_estimates(
                        estimates.estimates[0].rows(), time_steps.size());
                    Eigen::MatrixXd sliced_true_values(
                        estimates.estimates[0].rows(), time_steps.size());

                    int i = 0;
                    for (int t : time_steps) {
                        sliced_estimates.col(i) = estimates.estimates[0].col(t);
                        sliced_true_values.col(i) = true_values.col(t);

                        ConfusionMatrix confusion_matrix =
                            this->get_confusion_matrix(
                                sliced_estimates.col(i),
                                sliced_true_values.col(i),
                                estimates.assignment[0]);

                        f1s(0, i) = this->get_score(confusion_matrix);
                        i++;
                    }

                    if (num_f1s > 1) {
                        ConfusionMatrix confusion_matrix =
                            this->get_confusion_matrix(sliced_estimates,
                                                       sliced_true_values,
                                                       estimates.assignment[0]);
                        f1s(0, num_f1s - 1) = this->get_score(confusion_matrix);
                    }

                    evaluation.evaluation = f1s;
                }
            }

            return evaluation;
        }

        double
        F1Score::get_score(const ConfusionMatrix& confusion_matrix) const {
            double precision = 0;
            if (confusion_matrix.true_positives +
                confusion_matrix.false_positives >
                0) {
                precision = (double)confusion_matrix.true_positives /
                            (confusion_matrix.true_positives +
                             confusion_matrix.false_positives);
            }

            double recall = 0;
            if (confusion_matrix.true_positives +
                confusion_matrix.false_negatives >
                0) {
                recall = (double)confusion_matrix.true_positives /
                         (confusion_matrix.true_positives +
                          confusion_matrix.false_negatives);
            }

            double f1_score = 0;
            if (precision > 0 and recall > 0) {
                f1_score = (2 * precision * recall) / (precision + recall);
            }

            return f1_score;
        }

        void F1Score::get_info(nlohmann::json& json) const {
            json["name"] = NAME;
            json["threshold"] = this->threshold;
            this->estimator->get_info(json["estimator"]);
        }

    } // namespace model
} // namespace tomcat
