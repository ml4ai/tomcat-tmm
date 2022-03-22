#include "TeamQualityDecayEstimator.h"

#include <fmt/format.h>

#include "asist/study3/ASISTStudy3MessageConverter.h"
#include "utils/EigenExtensions.h"

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        TeamQualityDecayEstimator::TeamQualityDecayEstimator(
            const std::shared_ptr<DynamicBayesNet>& model,
            FREQUENCY_TYPE frequency_type,
            const nlohmann::json& json_config) {

            this->model = model;
            this->inference_horizon = 1;
            this->estimates.label = NAME;
            this->frequency_type = frequency_type;

            this->player_number = json_config["player_number"];

            this->prepare();
        }

        TeamQualityDecayEstimator::~TeamQualityDecayEstimator() {}

        //----------------------------------------------------------------------
        // Copy & Move constructors/assignments
        //----------------------------------------------------------------------
        TeamQualityDecayEstimator::TeamQualityDecayEstimator(
            const TeamQualityDecayEstimator& final_score) {
            SamplerEstimator::copy(final_score);
        }

        TeamQualityDecayEstimator& TeamQualityDecayEstimator::operator=(
            const TeamQualityDecayEstimator& final_score) {
            SamplerEstimator::copy(final_score);
            return *this;
        }

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------

        void TeamQualityDecayEstimator::prepare() {
            // Estimated team quality decay
            this->estimates.estimates = vector<Eigen::MatrixXd>(1);
        }

        string TeamQualityDecayEstimator::get_name() const { return NAME; }

        void TeamQualityDecayEstimator::estimate(
            const EvidenceSet& new_data,
            const EvidenceSet& particles,
            const EvidenceSet& projected_particles,
            const EvidenceSet& marginals,
            int data_point_idx,
            int time_step,
            ParticleFilter& filter) {

            const auto& dbn =
                dynamic_pointer_cast<DynamicBayesNet>(this->model);

            // In the online estimation, the new_data will be processed per time
            // step, so there will be only one observation available. In the
            // offline estimation, data from all the time steps will be
            // available, so we can index the time step directly.
            int real_time_step = time_step;
            time_step = min(time_step, new_data.get_time_steps() - 1);

            string node_map_section_label =
                ASISTStudy3MessageConverter::get_player_variable_label(
                    ASISTStudy3MessageConverter::MAP_SECTION,
                    this->player_number);
            int current_map_section = new_data[node_map_section_label].at(
                0, data_point_idx, time_step);

            string node_elapsed_seconds_map_section_label =
                ASISTStudy3MessageConverter::get_player_variable_label(
                    fmt::format("{}{}",
                                ASISTStudy3MessageConverter::
                                    ELAPSED_SECONDS_MAP_SECTION,
                                current_map_section + 1),
                    this->player_number);
            int current_elapsed_seconds =
                new_data[node_elapsed_seconds_map_section_label].at(
                    0, data_point_idx, time_step);

            Eigen::MatrixXd elapsed_sections(1, SECONDS_IN_SECTION);
            for (int i = 0; i < SECONDS_IN_SECTION; i++) {
                elapsed_sections(0, i) = ++current_elapsed_seconds;
            }

            EvidenceSet possible_future;
            possible_future.add_data(node_elapsed_seconds_map_section_label,
                                     elapsed_sections);
            auto [new_particles, new_marginals] =
                filter.forward_particles(SECONDS_IN_SECTION, possible_future);

            // Check the quality that changes the most and see if it's smaller
            // than the most likely current quality.
            int team_quality_cardinality =
                dbn->get_cardinality_of("TeamQuality");
            Eigen::VectorXd projected_team_quality =
                this->calculate_probabilities_from_samples(
                    new_particles["TeamQuality"](data_point_idx, 0)
                        .col(SECONDS_IN_SECTION - 1),
                    team_quality_cardinality);
            Eigen::VectorXd current_team_quality =
                this->calculate_probabilities_from_samples(
                    particles["TeamQuality"](data_point_idx, 0).col(0),
                    team_quality_cardinality);
            auto team_quality_change =
                (projected_team_quality - current_team_quality).array() /
                (current_team_quality.array() + EPSILON);

            int most_likely_quality = 0;
            for (int i = 1; i < current_team_quality.size(); i++) {
                if (current_team_quality(i) >
                    current_team_quality(most_likely_quality)) {
                    most_likely_quality = i;
                }
            }

            double quality_decay;
            if (most_likely_quality == 2) {
                // Estimate is by how much the quality 2 (good) decreased
                quality_decay = -team_quality_change(2);
            }
            else {
                // Estimate is by how much the quality 0 (bad) increased
                quality_decay = team_quality_change(0);
            }

            this->update_estimates(
                0, data_point_idx, real_time_step, quality_decay);
        }

    } // namespace model
} // namespace tomcat
