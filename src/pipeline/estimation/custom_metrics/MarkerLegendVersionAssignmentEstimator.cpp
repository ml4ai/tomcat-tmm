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
                const std::shared_ptr<DynamicBayesNet>& model) {

            this->model = model;
            this->inference_horizon = 0;
            this->estimates.label = NAME;
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
            const EvidenceSet& particles,
            const EvidenceSet& projected_particles,
            const EvidenceSet& marginals,
            int data_point_idx,
            int time_step) {

            for (int t = 0; t < marginals.get_time_steps(); t++) {
                vector<Eigen::VectorXd> marker_version_samples(3);

                for (int player_number = 0; player_number < 3;
                     player_number++) {
                    string marker_version_node_label =
                        MessageConverter::get_player_variable_label(
                            ASISTMultiPlayerMessageConverter::
                                PLAYER_MARKER_LEGEND_VERSION_LABEL,
                            player_number + 1);
                    marker_version_samples[player_number] =
                        marginals[marker_version_node_label](0, 0).col(
                            time_step);
                }

                Eigen::VectorXd valid_assignments(3);
                valid_assignments[0] =
                    marker_version_samples
                        [0][ASISTMultiPlayerMessageConverter::MARKER_LEGEND_B] *
                    marker_version_samples
                        [1][ASISTMultiPlayerMessageConverter::MARKER_LEGEND_A] *
                    marker_version_samples
                        [2][ASISTMultiPlayerMessageConverter::MARKER_LEGEND_A];

                valid_assignments[1] =
                    marker_version_samples
                        [0][ASISTMultiPlayerMessageConverter::MARKER_LEGEND_A] *
                    marker_version_samples
                        [1][ASISTMultiPlayerMessageConverter::MARKER_LEGEND_B] *
                    marker_version_samples
                        [2][ASISTMultiPlayerMessageConverter::MARKER_LEGEND_A];

                valid_assignments[2] =
                    marker_version_samples
                        [0][ASISTMultiPlayerMessageConverter::MARKER_LEGEND_A] *
                    marker_version_samples
                        [1][ASISTMultiPlayerMessageConverter::MARKER_LEGEND_A] *
                    marker_version_samples
                        [2][ASISTMultiPlayerMessageConverter::MARKER_LEGEND_B];

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
