#include "TrainingFrequencyEstimator.h"

#include <iostream>

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        TrainingFrequencyEstimator::TrainingFrequencyEstimator(
            const shared_ptr<DynamicBayesNet>& model,
            int inference_horizon,
            const std::string& node_label,
            const Eigen::VectorXd& assignment)
            : PGMEstimator(model, inference_horizon, node_label, assignment) {

            if (inference_horizon > 0 && assignment.size() == 0) {
                throw TomcatModelException(
                    "An assignment must be given for estimations with "
                    "inference horizon greater than 0.");
            }
        }

        TrainingFrequencyEstimator::~TrainingFrequencyEstimator() {}

        //----------------------------------------------------------------------
        // Copy & Move constructors/assignments
        //----------------------------------------------------------------------
        TrainingFrequencyEstimator::TrainingFrequencyEstimator(
            const TrainingFrequencyEstimator& estimator) {
            PGMEstimator::copy(estimator);
        }

        TrainingFrequencyEstimator& TrainingFrequencyEstimator::operator=(
            const TrainingFrequencyEstimator& estimator) {
            PGMEstimator::copy(estimator);
            return *this;
        }

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        void TrainingFrequencyEstimator::estimate(const EvidenceSet& new_data) {
            if (!this->training_data.has_data_for(this->estimates.label)) {
                throw TomcatModelException(
                    "No training data was provided to the node " +
                    this->estimates.label);
            }

            vector<Eigen::VectorXd> assignments;

            if (this->estimates.assignment.size() == 0) {
                // If no assignment was provided, we compute the
                // estimates for each one of the possible assignments
                // the node can have.
                int cardinality =
                    dynamic_pointer_cast<DynamicBayesNet>(this->model)
                        ->get_cardinality_of(this->estimates.label);

                for (int assignment = 0; assignment < cardinality;
                     assignment++) {
                    assignments.push_back(
                        Eigen::VectorXd::Constant(1, assignment));
                }
            }
            else {
                assignments.push_back(this->estimates.assignment);
            }

            int i = 0;
            for (const auto& assignment : assignments) {
                Eigen::MatrixXd logical_data_in_horizon =
                    this->training_data.get_observations_in_window_for(
                        this->estimates.label,
                        assignment,
                        this->inference_horizon);

                Eigen::MatrixXd estimates(new_data.get_num_data_points(),
                                          logical_data_in_horizon.cols());
                estimates.row(0) = logical_data_in_horizon.colwise().mean();
                estimates = estimates.row(0).replicate(
                    new_data.get_num_data_points(), 1);

                this->estimates.estimates[i++] = estimates;
            }
        }

        string TrainingFrequencyEstimator::get_name() const { return NAME; }

    } // namespace model
} // namespace tomcat
