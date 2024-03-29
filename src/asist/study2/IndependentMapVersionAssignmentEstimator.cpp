#include "IndependentMapVersionAssignmentEstimator.h"

#include "asist/study2/ASISTMultiPlayerMessageConverter.h"
#include "utils/EigenExtensions.h"

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        IndependentMapVersionAssignmentEstimator::IndependentMapVersionAssignmentEstimator(
            const std::shared_ptr<DynamicBayesNet>& model,
            FREQUENCY_TYPE frequency_type) {

            this->model = model;
            this->inference_horizon = 0;
            this->estimates.label = "MapVersionAssignment";
            this->frequency_type = frequency_type;
            this->prepare();
        }

        IndependentMapVersionAssignmentEstimator::~IndependentMapVersionAssignmentEstimator() {}

        //----------------------------------------------------------------------
        // Copy & Move constructors/assignments
        //----------------------------------------------------------------------
        IndependentMapVersionAssignmentEstimator::IndependentMapVersionAssignmentEstimator(
            const IndependentMapVersionAssignmentEstimator& final_score) {
            SamplerEstimator::copy(final_score);
        }

        IndependentMapVersionAssignmentEstimator& IndependentMapVersionAssignmentEstimator::operator=(
            const IndependentMapVersionAssignmentEstimator& final_score) {
            SamplerEstimator::copy(final_score);
            return *this;
        }

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------

        void IndependentMapVersionAssignmentEstimator::prepare() {
            this->estimates.estimates =
                vector<Eigen::MatrixXd>(6); // Possible assignments
        }

        string IndependentMapVersionAssignmentEstimator::get_name() const { return NAME; }

        void IndependentMapVersionAssignmentEstimator::estimate(
            const EvidenceSet& new_data,
            const EvidenceSet& particles,
            const EvidenceSet& projected_particles,
            const EvidenceSet& marginals,
            int data_point_idx,
            int time_step,
            ParticleFilter& filter) {

            for(int t = 0; t < marginals.get_time_steps(); t++) {
                vector<Eigen::VectorXd> map_version_samples(3);

                for (int player_number = 0; player_number < 3;
                     player_number++) {
                    string map_version_node_label =
                        MessageConverter::get_player_variable_label(
                            ASISTMultiPlayerMessageConverter::
                            PLAYER_MAP_VERSION_LABEL,
                            player_number + 1);
                    map_version_samples[player_number] =
                        marginals[map_version_node_label](0, 0).col(t);
                }

                Eigen::VectorXd valid_assignments(6);
                valid_assignments[0] =
                    map_version_samples
                    [0][ASISTMultiPlayerMessageConverter::SECTIONS_2N4] *
                    map_version_samples
                    [1][ASISTMultiPlayerMessageConverter::SECTIONS_3N4] *
                    map_version_samples
                    [2][ASISTMultiPlayerMessageConverter::SECTIONS_6N4];

                valid_assignments[1] =
                    map_version_samples
                    [0][ASISTMultiPlayerMessageConverter::SECTIONS_2N4] *
                    map_version_samples
                    [1][ASISTMultiPlayerMessageConverter::SECTIONS_6N4] *
                    map_version_samples
                    [2][ASISTMultiPlayerMessageConverter::SECTIONS_3N4];

                valid_assignments[2] =
                    map_version_samples
                    [0][ASISTMultiPlayerMessageConverter::SECTIONS_3N4] *
                    map_version_samples
                    [1][ASISTMultiPlayerMessageConverter::SECTIONS_2N4] *
                    map_version_samples
                    [2][ASISTMultiPlayerMessageConverter::SECTIONS_6N4];

                valid_assignments[3] =
                    map_version_samples
                    [0][ASISTMultiPlayerMessageConverter::SECTIONS_3N4] *
                    map_version_samples
                    [1][ASISTMultiPlayerMessageConverter::SECTIONS_6N4] *
                    map_version_samples
                    [2][ASISTMultiPlayerMessageConverter::SECTIONS_2N4];

                valid_assignments[4] =
                    map_version_samples
                    [0][ASISTMultiPlayerMessageConverter::SECTIONS_6N4] *
                    map_version_samples
                    [1][ASISTMultiPlayerMessageConverter::SECTIONS_2N4] *
                    map_version_samples
                    [2][ASISTMultiPlayerMessageConverter::SECTIONS_3N4];

                valid_assignments[5] =
                    map_version_samples
                    [0][ASISTMultiPlayerMessageConverter::SECTIONS_6N4] *
                    map_version_samples
                    [1][ASISTMultiPlayerMessageConverter::SECTIONS_3N4] *
                    map_version_samples
                    [2][ASISTMultiPlayerMessageConverter::SECTIONS_2N4];

                valid_assignments.array() /= valid_assignments.sum();

                for (int i = 0; i < 6; i++) {
                    this->update_estimates(
                        i, data_point_idx, time_step, valid_assignments[i]);
                }

                time_step++;
            }
        }

    } // namespace model
} // namespace tomcat
