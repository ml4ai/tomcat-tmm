#include "TeamQualityDecayEstimator.h"

#include "converter/ASISTStudy3MessageConverter.h"
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
            this->inference_horizon = 0;
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
            // Estimated final score
            this->estimates.estimates = vector<Eigen::MatrixXd>(1);

            // Average number of regular rescues
            // Std of regular rescues
            // Average number of critical rescues
            // Std of critical rescues
            this->estimates.custom_data = vector<Eigen::MatrixXd>(4);
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

            string node_map_section_label =
                ASISTStudy3MessageConverter::get_player_variable_label(
                    ASISTStudy3MessageConverter::MAP_SECTION,
                    this->player_number);
            int current_map_section =
                new_data[node_map_section_label](data_point_idx, time_step);

            string node_elapsed_seconds_map_section_label =
                ASISTStudy3MessageConverter::get_player_variable_label(
                    fmt::format("{}{}",
                                ASISTStudy3MessageConverter::
                                    ELAPSED_SECONDS_MAP_SECTION,
                                current_map_section),
                    this->player_number);
            int current_elapsed_seconds =
                new_data[node_elapsed_seconds_map_section_label](data_point_idx,
                                                                 time_step);

            Eigen::MatrixXd elapsed_sections(1, SECONDS_IN_SECTION);
            for (int i = 0; i < SECONDS_IN_SECTION; i++) {
                if (current_map_section == 0) {
                    elapsed_sections(0, i) = ++current_elapsed_seconds;
                }
            }

            EvidenceSet possible_future;
            possible_future.add_data(node_elapsed_seconds_map_section_label,
                                     elapsed_sections);
            EvidenceSet estimates =
                filter.forward_particles(SECONDS_IN_SECTION, possible_future);

            estimates["TeamQuality"]
        }

        Eigen::MatrixXi TeamQualityDecayEstimator::get_projected_rescues(
            const EvidenceSet& particles,
            const EvidenceSet& projected_particles) const {

            Eigen::MatrixXi rescues(particles.get_num_data_points(), 2);

            if (!projected_particles.empty()) {
                for (int i = 0; i < this->num_players; i++) {
                    string task_node_label =
                        MessageConverter::get_player_variable_label(
                            ASISTMultiPlayerMessageConverter::PLAYER_TASK_LABEL,
                            i + 1);

                    // Last sampled state
                    Eigen::VectorXd last_task_samples =
                        particles[task_node_label](0, 0).col(0);

                    Eigen::VectorXi last_regular_samples =
                        (last_task_samples.array() ==
                         ASISTMultiPlayerMessageConverter::SAVING_REGULAR)
                            .cast<int>();
                    Eigen::VectorXi last_critical_samples =
                        (last_task_samples.array() ==
                         ASISTMultiPlayerMessageConverter::SAVING_CRITICAL)
                            .cast<int>();

                    // Projection
                    Eigen::MatrixXd task_samples =
                        projected_particles[task_node_label](0, 0);

                    int rows = projected_particles.get_num_data_points();
                    int cols = projected_particles.get_time_steps();

                    // Regular
                    Eigen::MatrixXi tmp =
                        (task_samples.array() ==
                         ASISTMultiPlayerMessageConverter::SAVING_REGULAR)
                            .block(0, 0, rows, cols - 1)
                            .cast<int>();

                    Eigen::MatrixXi regular_rescue_samples(rows, cols);
                    regular_rescue_samples.col(0) = last_regular_samples;
                    regular_rescue_samples.block(0, 1, rows, cols - 1) = tmp;

                    Eigen::MatrixXi not_regular_rescue_next_samples =
                        (task_samples.array() !=
                         ASISTMultiPlayerMessageConverter::SAVING_REGULAR)
                            .cast<int>();

                    Eigen::MatrixXi finished_regular_rescues =
                        (regular_rescue_samples.array() *
                         not_regular_rescue_next_samples.array());

                    // Critical
                    tmp = (task_samples.array() ==
                           ASISTMultiPlayerMessageConverter::SAVING_CRITICAL)
                              .matrix()
                              .block(0, 0, rows, cols - 1)
                              .cast<int>();

                    Eigen::MatrixXi critical_rescue_samples(rows, cols);
                    critical_rescue_samples.col(0) = last_critical_samples;
                    critical_rescue_samples.block(0, 1, rows, cols - 1) = tmp;

                    Eigen::MatrixXi not_critical_rescue_next_samples =
                        (task_samples.array() !=
                         ASISTMultiPlayerMessageConverter::SAVING_CRITICAL)
                            .cast<int>();

                    Eigen::MatrixXi finished_critical_rescues =
                        (critical_rescue_samples.array() *
                         not_critical_rescue_next_samples.array());

                    rescues.col(0) = finished_regular_rescues.rowwise().sum();
                    rescues.col(1) = finished_critical_rescues.rowwise().sum();
                }
            }

            return rescues;
        }

    } // namespace model
} // namespace tomcat
