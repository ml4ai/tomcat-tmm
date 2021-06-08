#include "FinalScore.h"

#include "converter/ASISTMultiPlayerMessageConverter.h"
#include "utils/EigenExtensions.h"

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        FinalScore::FinalScore(const std::shared_ptr<DynamicBayesNet>& model) {
            this->model = model;
            this->inference_horizon = -1;
            this->estimates.label = "FinalScore";
            this->estimates.estimates =
                vector<Eigen::MatrixXd>(2); // avg score and std
        }

        FinalScore::~FinalScore() {}

        //----------------------------------------------------------------------
        // Copy & Move constructors/assignments
        //----------------------------------------------------------------------
        FinalScore::FinalScore(const FinalScore& final_score) {
            SamplerEstimator::copy(final_score);
        }

        FinalScore& FinalScore::operator=(const FinalScore& final_score) {
            SamplerEstimator::copy(final_score);
            return *this;
        }

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------

        void FinalScore::prepare() {
            this->estimates.estimates =
                vector<Eigen::MatrixXd>(2); // avg score and std
            this->current_score = Eigen::VectorXd(0);
            this->last_regular_samples = Eigen::VectorXi(0);
            this->last_critical_samples = Eigen::VectorXi(0);
        }

        string FinalScore::get_name() const { return "final_score"; }

        void FinalScore::estimate(const EvidenceSet& particles,
                                  const EvidenceSet& projected_particles,
                                  int data_point_idx,
                                  int time_step) {

            Eigen::VectorXd task_samples =
                particles[ASISTMultiPlayerMessageConverter::TASK](0, 0).col(0);

            this->current_score = this->get_current_score(particles);

            this->last_regular_samples =
                (task_samples.array() ==
                 ASISTMultiPlayerMessageConverter::SAVING_REGULAR)
                    .cast<int>();
            this->last_critical_samples =
                (task_samples.array() ==
                 ASISTMultiPlayerMessageConverter::SAVING_CRITICAL)
                    .cast<int>();

            Eigen::VectorXd projected_score =
                this->get_projected_score(projected_particles);

            Eigen::VectorXd estimated_score =
                this->current_score.array() + projected_score.array();

            double avg_estimated_score = estimated_score.mean();
            double std_estimated_score = sqrt(
                (estimated_score.array() - avg_estimated_score).square().sum() /
                (estimated_score.size() - 1));

            this->update_estimates(
                0, data_point_idx, time_step, avg_estimated_score);
            this->update_estimates(
                1, data_point_idx, time_step, std_estimated_score);
        }

        Eigen::VectorXd
        FinalScore::get_current_score(const EvidenceSet& particles) const {
            Eigen::VectorXd score =
                Eigen::VectorXd::Zero(particles.get_num_data_points());

            if (this->last_regular_samples.size() != 0) {
                Eigen::VectorXd task_samples =
                    particles[ASISTMultiPlayerMessageConverter::TASK](0, 0).col(
                        0);

                Eigen::VectorXi finished_regular_rescues =
                    this->last_regular_samples.array() *
                    (task_samples.array() !=
                     ASISTMultiPlayerMessageConverter::SAVING_REGULAR)
                        .cast<int>();

                score = this->current_score.array() +
                        (REGULAR_SCORE * finished_regular_rescues.array())
                            .cast<double>();

                Eigen::VectorXi finished_critical_rescues =
                    this->last_critical_samples.array() *
                    (task_samples.array() !=
                     ASISTMultiPlayerMessageConverter::SAVING_CRITICAL)
                        .cast<int>();
                score = score.array() +
                        (CRITICAL_SCORE * finished_critical_rescues.array())
                            .cast<double>();
            }

            return score;
        }

        Eigen::VectorXd FinalScore::get_projected_score(
            const EvidenceSet& projected_particles) const {

            Eigen::VectorXd projected_score =
                Eigen::VectorXd::Zero(this->last_regular_samples.size());

            if (!projected_particles.empty()) {
                Eigen::MatrixXd task_samples =
                    projected_particles[ASISTMultiPlayerMessageConverter::TASK](
                        0, 0);

                int rows = projected_particles.get_num_data_points();
                int cols = projected_particles.get_time_steps();

                // Regular
                Eigen::MatrixXi tmp =
                    (task_samples.array() ==
                     ASISTMultiPlayerMessageConverter::SAVING_REGULAR)
                        .block(0, 0, rows, cols - 1)
                        .cast<int>();

                Eigen::MatrixXi regular_rescue_samples(rows, cols);
                regular_rescue_samples.col(0) = this->last_regular_samples;
                regular_rescue_samples.block(0, 1, rows, cols - 1) = tmp;

                Eigen::MatrixXi not_regular_rescue_next_samples =
                    (task_samples.array() !=
                     ASISTMultiPlayerMessageConverter::SAVING_REGULAR)
                        .cast<int>();

                Eigen::MatrixXi finished_regular_rescues =
                    (regular_rescue_samples.array() *
                     not_regular_rescue_next_samples.array());

                projected_score =
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
                critical_rescue_samples.col(0) = this->last_critical_samples;
                critical_rescue_samples.block(0, 1, rows, cols - 1) = tmp;

                Eigen::MatrixXi not_critical_rescue_next_samples =
                    (task_samples.array() !=
                     ASISTMultiPlayerMessageConverter::SAVING_CRITICAL)
                        .cast<int>();

                Eigen::MatrixXi finished_critical_rescues =
                    (critical_rescue_samples.array() *
                     not_critical_rescue_next_samples.array());

                projected_score =
                    projected_score.array() +
                    (CRITICAL_SCORE * finished_critical_rescues.array())
                        .rowwise()
                        .sum()
                        .cast<double>();
            }

            return projected_score;
        }

        //----------------------------------------------------------------------
        // Getters & Setters
        //----------------------------------------------------------------------

    } // namespace model
} // namespace tomcat
