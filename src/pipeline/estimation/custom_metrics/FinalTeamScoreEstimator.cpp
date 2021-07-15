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
            const std::shared_ptr<DynamicBayesNet>& model) {

            this->model = model;
            this->inference_horizon = -1;
            this->estimates.label = NAME;
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
            this->estimates.estimates =
                vector<Eigen::MatrixXd>(2); // avg score and std
            this->current_score = Eigen::VectorXd(0);

            this->last_regular_samples =
                vector<Eigen::VectorXi>(this->num_players);
            this->last_critical_samples =
                vector<Eigen::VectorXi>(this->num_players);
            for (int i = 0; i < this->num_players; i++) {
                this->last_regular_samples[i] = Eigen::VectorXi(0);
                this->last_critical_samples[i] = Eigen::VectorXi(0);
            }
        }

        string FinalTeamScoreEstimator::get_name() const {
            return "final_team_score";
        }

        void FinalTeamScoreEstimator::estimate(
            const EvidenceSet& particles,
            const EvidenceSet& projected_particles,
            const EvidenceSet& marginals,
            int data_point_idx,
            int time_step) {

            for (int i = 0; i < this->num_players; i++) {
                string task_node_label =
                    MessageConverter::get_player_variable_label(
                        ASISTMultiPlayerMessageConverter::PLAYER_TASK_LABEL, i + 1);

                Eigen::VectorXd task_samples =
                    particles[task_node_label](0, 0).col(0);

                this->last_regular_samples[i] =
                    (task_samples.array() ==
                     ASISTMultiPlayerMessageConverter::SAVING_REGULAR)
                        .cast<int>();
                this->last_critical_samples[i] =
                    (task_samples.array() ==
                     ASISTMultiPlayerMessageConverter::SAVING_CRITICAL)
                        .cast<int>();
            }

            Eigen::VectorXd score =
                Eigen::VectorXd::Zero(particles.get_num_data_points());

            int avg_estimated_score = 0;
            double std_estimated_score = 0;
            if (time_step == 0) {
                this->current_score =
                    Eigen::VectorXd::Zero(particles.get_num_data_points());
            }
            else {
                this->current_score = this->get_current_score(particles);
                Eigen::VectorXd projected_score =
                    this->get_projected_score(projected_particles);
                Eigen::VectorXd estimated_score =
                    this->current_score.array() + projected_score.array();

                avg_estimated_score = estimated_score.mean();
                std_estimated_score =
                    sqrt((estimated_score.array() - avg_estimated_score)
                             .square()
                             .sum() /
                         (estimated_score.size() - 1));
            }

            this->update_estimates(
                0, data_point_idx, time_step, avg_estimated_score);
            this->update_estimates(
                1, data_point_idx, time_step, std_estimated_score);
        }

        Eigen::VectorXd FinalTeamScoreEstimator::get_current_score(
            const EvidenceSet& particles) const {
            Eigen::VectorXd score =
                Eigen::VectorXd::Zero(particles.get_num_data_points());

            for (int i = 0; i < this->num_players; i++) {
                string task_node_label =
                    MessageConverter::get_player_variable_label(
                        ASISTMultiPlayerMessageConverter::PLAYER_TASK_LABEL, i + 1);

                Eigen::VectorXd task_samples =
                    particles[task_node_label](0, 0).col(0);

                Eigen::VectorXi finished_regular_rescues =
                    this->last_regular_samples[i].array() *
                    (task_samples.array() !=
                     ASISTMultiPlayerMessageConverter::SAVING_REGULAR)
                        .cast<int>();

                score = score.array() + this->current_score.array() +
                        (REGULAR_SCORE * finished_regular_rescues.array())
                            .cast<double>();

                Eigen::VectorXi finished_critical_rescues =
                    this->last_critical_samples[i].array() *
                    (task_samples.array() !=
                     ASISTMultiPlayerMessageConverter::SAVING_CRITICAL)
                        .cast<int>();
                score = score.array() +
                        (CRITICAL_SCORE * finished_critical_rescues.array())
                            .cast<double>();
            }

            return score;
        }

        Eigen::VectorXd FinalTeamScoreEstimator::get_projected_score(
            const EvidenceSet& projected_particles) const {

            Eigen::VectorXd projected_score =
                Eigen::VectorXd::Zero(this->last_regular_samples[0].size());

            if (!projected_particles.empty()) {
                for (int i = 0; i < this->num_players; i++) {
                    string task_node_label =
                        MessageConverter::get_player_variable_label(
                            ASISTMultiPlayerMessageConverter::PLAYER_TASK_LABEL,
                            i + 1);

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
                    regular_rescue_samples.col(0) =
                        this->last_regular_samples[i];
                    regular_rescue_samples.block(0, 1, rows, cols - 1) = tmp;

                    Eigen::MatrixXi not_regular_rescue_next_samples =
                        (task_samples.array() !=
                         ASISTMultiPlayerMessageConverter::SAVING_REGULAR)
                            .cast<int>();

                    Eigen::MatrixXi finished_regular_rescues =
                        (regular_rescue_samples.array() *
                         not_regular_rescue_next_samples.array());

                    projected_score.array() +=
                        (REGULAR_SCORE * finished_regular_rescues.array())
                            .rowwise()
                            .sum()
                            .cast<double>();

                    // Critical
                    tmp = (task_samples.array() ==
                           ASISTMultiPlayerMessageConverter::SAVING_CRITICAL)
                              .matrix()
                              .block(0, 0, rows, cols - 1)
                              .cast<int>();

                    Eigen::MatrixXi critical_rescue_samples(rows, cols);
                    critical_rescue_samples.col(0) =
                        this->last_critical_samples[i];
                    critical_rescue_samples.block(0, 1, rows, cols - 1) = tmp;

                    Eigen::MatrixXi not_critical_rescue_next_samples =
                        (task_samples.array() !=
                         ASISTMultiPlayerMessageConverter::SAVING_CRITICAL)
                            .cast<int>();

                    Eigen::MatrixXi finished_critical_rescues =
                        (critical_rescue_samples.array() *
                         not_critical_rescue_next_samples.array());

                    projected_score.array() +=
                        (CRITICAL_SCORE * finished_critical_rescues.array())
                            .rowwise()
                            .sum()
                            .cast<double>();
                }
            }

            return projected_score;
        }

    } // namespace model
} // namespace tomcat
