#include "RMSE.h"

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        RMSE::RMSE(const shared_ptr<Estimator>& estimator,
                   FREQUENCY_TYPE frequency_type)
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
                // Get matrix of true observations.
                Eigen::MatrixXd true_values = test_data[evaluation.label](0, 0);

                int cols = estimates.estimates[0].cols();
                int first_valid_time_step =
                    EvidenceSet::get_first_time_with_observation(
                        test_data[evaluation.label]);
                vector<int> time_steps;
                if (this->frequency_type == last) {
                    time_steps.push_back(cols - 1);
                }
                else if (this->frequency_type == fixed) {
                    time_steps = this->fixed_steps;
                }

                if (this->frequency_type == all) {
                    double rmse =
                        this->get_score(true_values, estimates.estimates[0]);
                    evaluation.evaluation =
                        Eigen::MatrixXd::Constant(1, 1, rmse);
                }
                else {
                    if (time_steps.empty())
                        return evaluation;

                    int num_rmses =
                        this->frequency_type == fixed && time_steps.size() > 1
                            ? time_steps.size() + 1
                            : 1;
                    Eigen::MatrixXd rmses = Eigen::MatrixXd(1, num_rmses);

                    Eigen::MatrixXd sliced_estimates(
                        estimates.estimates[0].rows(), time_steps.size());
                    Eigen::MatrixXd sliced_true_values(
                        estimates.estimates[0].rows(), time_steps.size());

                    int i = 0;
                    for (int t : time_steps) {
                        sliced_estimates.col(i) = estimates.estimates[0].col(t);
                        sliced_true_values.col(i) = true_values.col(t);

                        rmses(0, i) = this->get_score(sliced_true_values.col(i),
                                                      sliced_estimates.col(i));
                        i++;
                    }

                    if (num_rmses > 1) {
                        rmses(0, num_rmses - 1) = this->get_score(
                            sliced_true_values, sliced_estimates);
                    }

                    evaluation.evaluation = rmses;
                }
            }

            return evaluation;
        }

        double RMSE::get_score(const Eigen::MatrixXd& true_values,
                               const Eigen::MatrixXd& estimated_values) const {
            double rmse = sqrt(
                (estimated_values.array() - true_values.array()).pow(2).sum() /
                true_values.size());

            return rmse;
        }

        void RMSE::get_info(nlohmann::json& json) const {
            json["name"] = NAME;
            json["threshold"] = this->threshold;
            this->estimator->get_info(json["estimator"]);
        }

    } // namespace model
} // namespace tomcat
