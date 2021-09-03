#include "MarkerLegendVersionAssignmentEstimator.h"

#include "converter/ASISTMultiPlayerMessageConverter.h"
#include "utils/EigenExtensions.h"

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        MarkerLegendVersionAssignmentEstimator::
            MarkerLegendVersionAssignmentEstimator(
                const std::shared_ptr<DynamicBayesNet>& model,
                FREQUENCY_TYPE frequency_type) {
            this->model = model;
            this->inference_horizon = 0;
            this->estimates.label = NAME;
            this->frequency_type = frequency_type;
            this->prepare();
        }

        MarkerLegendVersionAssignmentEstimator::
            ~MarkerLegendVersionAssignmentEstimator() {}

        //----------------------------------------------------------------------
        // Copy & Move constructors/assignments
        //----------------------------------------------------------------------
        MarkerLegendVersionAssignmentEstimator::
            MarkerLegendVersionAssignmentEstimator(
                const MarkerLegendVersionAssignmentEstimator& final_score) {
            SamplerEstimator::copy(final_score);
        }

        MarkerLegendVersionAssignmentEstimator&
        MarkerLegendVersionAssignmentEstimator::operator=(
            const MarkerLegendVersionAssignmentEstimator& final_score) {
            SamplerEstimator::copy(final_score);
            return *this;
        }

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------

        void MarkerLegendVersionAssignmentEstimator::prepare() {
            this->estimates.estimates =
                vector<Eigen::MatrixXd>(3); // Possible assignments
        }

        string MarkerLegendVersionAssignmentEstimator::get_name() const {
            return NAME;
        }

        void MarkerLegendVersionAssignmentEstimator::estimate(
            const EvidenceSet& new_data,
            const EvidenceSet& particles,
            const EvidenceSet& projected_particles,
            const EvidenceSet& marginals,
            int data_point_idx,
            int time_step) {

            for (int t = 0; t < marginals.get_time_steps(); t++) {
                vector<Eigen::VectorXd> legend_version_samples(3);

                for (int player_number = 0; player_number < 3;
                     player_number++) {
                    string legend_version_node_label =
                        MessageConverter::get_player_variable_label(
                            ASISTMultiPlayerMessageConverter::
                            PLAYER_MARKER_LEGEND_VERSION_LABEL,
                            player_number + 1);
                    legend_version_samples[player_number] =
                        particles[legend_version_node_label](0, 0).col(t);
                }

                Eigen::VectorXd valid_assignments = Eigen::VectorXd::Zero(3);
                for (int i = 0; i < legend_version_samples[0].rows(); i++) {
                    if (legend_version_samples[0][i] ==
                        ASISTMultiPlayerMessageConverter::MARKER_LEGEND_B) {
                        if (legend_version_samples[1][i] ==
                            ASISTMultiPlayerMessageConverter::MARKER_LEGEND_A) {
                            if (legend_version_samples[2][i] ==
                                ASISTMultiPlayerMessageConverter::MARKER_LEGEND_A) {
                                // P1 gets B, P2 gets A, P3 gets A
                                valid_assignments[0] += 1;
                            }
                        }
                    }
                    else {
                        if (legend_version_samples[1][i] ==
                            ASISTMultiPlayerMessageConverter::MARKER_LEGEND_B) {
                            if (legend_version_samples[2][i] ==
                                ASISTMultiPlayerMessageConverter::MARKER_LEGEND_A) {
                                // P1 gets A, P2 gets B, P3 gets A
                                valid_assignments[1] += 1;
                            }

                        } else {
                            if (legend_version_samples[2][i] ==
                                ASISTMultiPlayerMessageConverter::MARKER_LEGEND_B) {
                                // P1 gets A, P2 gets A, P3 gets B
                                valid_assignments[2] += 1;
                            }
                        }
                    }
                }

                valid_assignments.array() /= valid_assignments.sum();

                for (int i = 0; i < 3; i++) {
                    this->update_estimates(
                        i, data_point_idx, time_step, valid_assignments[i]);
                }

                time_step++;
            }
        }

    } // namespace model
} // namespace tomcat
