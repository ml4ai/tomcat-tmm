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
                    double num_correct_cases =
                        confusion_matrices[i].diagonal().sum();
                    double num_cases = confusion_matrices[i].sum();
                    double accuracy = num_correct_cases / num_cases;

                    evaluation.evaluation(0, i) = accuracy;
                    evaluation.confusion_matrix.block(
                        0,
                        i * confusion_matrices[i].cols(),
                        confusion_matrices[i].rows(),
                        confusion_matrices[i].cols()) = confusion_matrices[i];
                }
            }

            return evaluation;
        }

        void Accuracy::get_info(nlohmann::json& json) const {
            json["name"] = NAME;
            json["threshold"] = this->threshold;
            this->estimator->get_info(json["estimator"]);
        }

    } // namespace model
} // namespace tomcat
