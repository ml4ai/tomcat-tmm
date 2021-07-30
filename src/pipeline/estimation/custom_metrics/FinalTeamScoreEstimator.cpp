#include "FinalTeamScoreEstimator.h"

#include "converter/ASISTMultiPlayerMessageConverter.h"
#include "utils/EigenExtensions.h"

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        FinalTeamScoreEstimator::FinalTeamScoreEstimator(
            const std::shared_ptr<DynamicBayesNet>& model,
            FREQUENCY_TYPE frequency_type) {

            this->model = model;
            this->inference_horizon = -1;
            this->estimates.label = NAME;
            this->frequency_type = frequency_type;
            this->prepare();

            // Determine the number of players in the model
            while (model->has_node_with_label(
                MessageConverter::get_player_variable_label(
                    ASISTMultiPlayerMessageConverter::PLAYER_ROLE_LABEL,
                    this->num_players + 1))) {
                this->num_players++;
            }
        }

        FinalTeamScoreEstimator::~FinalTeamScoreEstimator() {}

        //----------------------------------------------------------------------
        // Copy & Move constructors/assignments
        //----------------------------------------------------------------------
        FinalTeamScoreEstimator::FinalTeamScoreEstimator(
            const FinalTeamScoreEstimator& final_score) {
            SamplerEstimator::copy(final_score);
        }

        FinalTeamScoreEstimator& FinalTeamScoreEstimator::operator=(
            const FinalTeamScoreEstimator& final_score) {
            SamplerEstimator::copy(final_score);
            return *this;
        }

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------

        void FinalTeamScoreEstimator::prepare() {
            // Estimated final score
            this->estimates.estimates = vector<Eigen::MatrixXd>(1);

            // Average number of regular rescues
            // Std of regular rescues
            // Average number of critical rescues
            // Std of critical rescues
            this->estimates.custom_data = vector<Eigen::MatrixXd>(4);
        }

        string FinalTeamScoreEstimator::get_name() const { return NAME; }

        void FinalTeamScoreEstimator::estimate(
            const EvidenceSet& new_data,
            const EvidenceSet& particles,
            const EvidenceSet& projected_particles,
            const EvidenceSet& marginals,
            int data_point_idx,
            int time_step) {

            int avg_estimated_score = 0;
            double std_estimated_score = 0;
            int current_score =
                new_data[ASISTMultiPlayerMessageConverter::TEAM_SCORE_LABEL](
                    0, 0)(data_point_idx, time_step);
            Eigen::MatrixXi projected_rescues =
                this->get_projected_rescues(particles, projected_particles);

            // Cap values larger than the maximum amount of victims in the game
            int n = projected_rescues.rows();
            for (int i = 0; i < n; i++) {
                projected_rescues(i, 0) =
                    min(projected_rescues(i, 0), MAX_REGULAR);
                projected_rescues(i, 1) =
                    min(projected_rescues(i, 1), MAX_CRITICAL);
            }

            Eigen::VectorXd avg_rescues =
                projected_rescues.cast<double>().colwise().mean();

            double std_regular_rescues =
                sqrt((projected_rescues.cast<double>().col(0).array() -
                      avg_rescues[0])
                         .array()
                         .square()
                         .sum() /
                     (n - 1));

            double std_critical_rescues =
                sqrt((projected_rescues.cast<double>().col(1).array() -
                      avg_rescues[1])
                         .array()
                         .square()
                         .sum() /
                     (n - 1));

            int projected_regular_score = ((int)avg_rescues[0]) * REGULAR_SCORE;
            int projected_critical_score =
                ((int)avg_rescues[1]) * CRITICAL_SCORE;
            int projected_score =
                projected_regular_score + projected_critical_score;
            int estimated_final_score = current_score + projected_score;

            this->update_estimates(
                0, data_point_idx, time_step, estimated_final_score);

            this->update_custom_data(
                0, data_point_idx, time_step, (int)avg_rescues[0]);
            this->update_custom_data(
                1, data_point_idx, time_step, std_regular_rescues);
            this->update_custom_data(
                2, data_point_idx, time_step, (int)avg_rescues[1]);
            this->update_custom_data(
                3, data_point_idx, time_step, std_critical_rescues);
        }

        Eigen::MatrixXi FinalTeamScoreEstimator::get_projected_rescues(
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
