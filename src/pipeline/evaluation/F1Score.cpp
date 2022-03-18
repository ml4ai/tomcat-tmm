#include "F1Score.h"

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        F1Score::F1Score(const shared_ptr<PGMEstimator>& estimator,
                         double threshold,
                         FREQUENCY_TYPE frequency_type,
                         bool macro)
            : Measure(estimator, threshold, frequency_type), macro(macro) {}

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
                vector<Eigen::MatrixXi> confusion_matrices =
                    this->get_confusion_matrices(test_data);

                evaluation.evaluation = Eigen::MatrixXd::Constant(
                    1, confusion_matrices.size(), NO_OBS);
                // Side by side matrices
                evaluation.confusion_matrix = Eigen::MatrixXi::Constant(
                    confusion_matrices[0].rows(),
                    confusion_matrices.size() * confusion_matrices[0].cols(),
                    NO_OBS);

                for (int i = 0; i < confusion_matrices.size(); i++) {
                    double f1 = this->get_score(confusion_matrices[i]);
                    evaluation.evaluation(0, i) = f1;
                    evaluation.confusion_matrix.block(
                        0,
                        i * confusion_matrices[i].cols(),
                        confusion_matrices[i].rows(),
                        confusion_matrices[i].cols()) = confusion_matrices[i];
                }
            }

            return evaluation;
        }

        double
        F1Score::get_score(const Eigen::MatrixXi& confusion_matrix) const {
            double f1;

            int num_classes = confusion_matrix.rows();
            if (num_classes == 2) {
                double tp = confusion_matrix(1, 1);
                double fp = confusion_matrix(1, 0);
                double fn = confusion_matrix(0, 1);

                double precision = 0 ? tp + fp == 0 : tp / (tp + fp);
                double recall = 0 ? tp + fn == 0 : tp / (tp + fn);
                f1 = (2 * precision * recall) / (precision + recall);
            }
            else {
                if (macro) {
                    // Per class
                    Eigen::VectorXd f1_per_class(num_classes);
                    for (int k = 0; k < num_classes; k++) {
                        double tp = confusion_matrix(k, k);
                        double fp = confusion_matrix.row(k).sum() -
                                    confusion_matrix(k, k);
                        double fn = confusion_matrix.col(k).sum() -
                                    confusion_matrix(k, k);

                        double precision = 0 ? tp + fp == 0 : tp / (tp + fp);
                        double recall = 0 ? tp + fn == 0 : tp / (tp + fn);
                        f1_per_class[k] =
                            (2 * precision * recall) / (precision + recall);
                    }

                    f1 = f1_per_class.mean();
                }
                else {
                    double tp = confusion_matrix.diagonal().sum();
                    double fp = confusion_matrix.sum() - tp;
                    double fn = fp;

                    double precision = 0 ? tp + fp == 0 : tp / (tp + fp);
                    double recall = 0 ? tp + fn == 0 : tp / (tp + fn);
                    f1 = (2 * precision * recall) / (precision + recall);
                }
            }

            return f1;
        }

        void F1Score::get_info(nlohmann::json& json) const {
            json["name"] = this->macro ? MACRO_NAME : MICRO_NAME;
            json["threshold"] = this->threshold;
            this->estimator->get_info(json["estimator"]);
        }

    } // namespace model
} // namespace tomcat
