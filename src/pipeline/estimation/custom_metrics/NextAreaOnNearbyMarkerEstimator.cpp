#include "NextAreaOnNearbyMarkerEstimator.h"

#include "converter/ASISTMultiPlayerMessageConverter.h"
#include "utils/EigenExtensions.h"

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        NextAreaOnNearbyMarkerEstimator::NextAreaOnNearbyMarkerEstimator(
            const std::shared_ptr<DynamicBayesNet>& model, int player_num)
            : player_num(player_num) {

            this->model = model;
            this->inference_horizon = 5;
            this->estimates.label =
                ASISTMultiPlayerMessageConverter::get_player_variable_label(
                    NAME, player_num);
            this->frequency_type = dynamic;
            this->prepare();
        }

        NextAreaOnNearbyMarkerEstimator::~NextAreaOnNearbyMarkerEstimator() {}

        //----------------------------------------------------------------------
        // Copy & Move constructors/assignments
        //----------------------------------------------------------------------
        NextAreaOnNearbyMarkerEstimator::NextAreaOnNearbyMarkerEstimator(
            const NextAreaOnNearbyMarkerEstimator& estimator) {
            SamplerEstimator::copy(estimator);
            this->player_num = estimator.player_num;
        }

        NextAreaOnNearbyMarkerEstimator&
        NextAreaOnNearbyMarkerEstimator::operator=(
            const NextAreaOnNearbyMarkerEstimator& estimator) {
            SamplerEstimator::copy(estimator);
            this->player_num = estimator.player_num;
            return *this;
        }

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------

        void NextAreaOnNearbyMarkerEstimator::prepare() {
            this->estimates.estimates = vector<Eigen::MatrixXd>(2);
            // Number of particles used in the estimation. Only particles in
            // which the samples from a marker block nearby are 0 (player left
            // the range of detection of a marker block)
            this->estimates.custom_data = vector<Eigen::MatrixXd>(1);
            this->prepare_for_the_next_data_point();
        }

        string NextAreaOnNearbyMarkerEstimator::get_name() const {
            return this->estimates.label;
        }

        void NextAreaOnNearbyMarkerEstimator::estimate(
            const EvidenceSet& new_data,
            const EvidenceSet& particles,
            const EvidenceSet& projected_particles,
            const EvidenceSet& marginals,
            int data_point_idx,
            int time_step) {

            int n = particles.get_num_data_points();
            Eigen::VectorXd areas = Eigen::VectorXd::Zero(2);
            vector<bool> out_of_marker_range(n, false);

            string nearby_marker_label =
                ASISTMultiPlayerMessageConverter::get_player_variable_label(
                    ASISTMultiPlayerMessageConverter::
                        OTHER_PLAYER_NEARBY_MARKER,
                    this->player_num);
            string area_label =
                ASISTMultiPlayerMessageConverter::get_player_variable_label(
                    ASISTMultiPlayerMessageConverter::PLAYER_AREA_LABEL,
                    this->player_num);

            int detected_marker =
                new_data[nearby_marker_label].at(0, data_point_idx, time_step);
            for (int i = 0; i < n; i++) {
                for (int t = 0; t < this->inference_horizon; t++) {
                    int nearby_marker =
                        projected_particles[nearby_marker_label].at(0, i, t);
                    if (nearby_marker != detected_marker) {
                        int area = projected_particles[area_label].at(0, i, t);
                        areas[area] += 1;
                        break;
                    }
                }
            }

            double num_valid_particles = areas.sum();
            areas.array() /= num_valid_particles;

            this->update_estimates(0, data_point_idx, time_step, areas[0]);
            this->update_estimates(1, data_point_idx, time_step, areas[1]);
            this->update_custom_data(
                0, data_point_idx, time_step, num_valid_particles);

            int current_nearby_marker =
                new_data[nearby_marker_label].at(0, data_point_idx, time_step);
        }

        bool NextAreaOnNearbyMarkerEstimator::is_event_triggered_at(
            int data_point, int time_step, const EvidenceSet& new_data) const {

            string nearby_marker_label =
                ASISTMultiPlayerMessageConverter::get_player_variable_label(
                    ASISTMultiPlayerMessageConverter::
                        OTHER_PLAYER_NEARBY_MARKER,
                    this->player_num);
            string area_label =
                ASISTMultiPlayerMessageConverter::get_player_variable_label(
                    ASISTMultiPlayerMessageConverter::PLAYER_AREA_LABEL,
                    this->player_num);

            bool entered_marker_range = false;
            int current_nearby_marker =
                new_data[nearby_marker_label].at(0, data_point, time_step);

            if (this->within_marker_range) {
                if (time_step == this->time_step_at_entrance) {
                    entered_marker_range = true;
                }
                else if (current_nearby_marker != 1 &&
                         current_nearby_marker != 2) {
                    this->within_marker_range = false;
                }
            }
            else {
                int current_area =
                    new_data[area_label].at(0, data_point, time_step);

                if (current_area == ASISTMultiPlayerMessageConverter::HALLWAY) {
                    if (current_nearby_marker == 1 ||
                        current_nearby_marker == 2) {

                        entered_marker_range = true;
                        this->within_marker_range = true;
                        this->time_step_at_entrance = time_step;
                    }
                }
            }

            return entered_marker_range;
        }

        bool NextAreaOnNearbyMarkerEstimator::is_binary_on_prediction() const {
            // A positive horizon is used in the estimates but we are not
            // constrained to a binary problem.
            return false;
        }

        void NextAreaOnNearbyMarkerEstimator::prepare_for_the_next_data_point()
            const {
            this->within_marker_range = false;
            this->time_step_at_entrance = -1;
        }

    } // namespace model
} // namespace tomcat
