#include "FinalScore.h"

#include <fmt/format.h>

#include "converter/ASISTMultiPlayerMessageConverter.h"

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        FinalScore::FinalScore() {}

        FinalScore::~FinalScore() {}

        //----------------------------------------------------------------------
        // Copy & Move constructors/assignments
        //----------------------------------------------------------------------
        FinalScore::FinalScore(const FinalScore& final_score) {
            CustomSamplingMetric::copy(final_score);
        }

        FinalScore& FinalScore::operator=(const FinalScore& final_score) {
            CustomSamplingMetric::copy(final_score);
            return *this;
        }

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------

        std::vector<double>
        FinalScore::calculate(const std::shared_ptr<Sampler>& sampler,
                              int time_step) const {
            int final_score = 0;

            int player_id = 1;
            int rows, cols = 0;
            const string& root_label = ASISTMultiPlayerMessageConverter::TASK;
            string node_label = fmt::format("{}P{}", root_label, player_id);

            while (sampler->get_model()->has_node_with_label(node_label)) {
                Eigen::MatrixXi regular_savings =
                    (sampler->get_samples(node_label)(0, 0).array() ==
                     ASISTMultiPlayerMessageConverter::SAVING_REGULAR)
                        .cast<int>();

                final_score += this->get_transitions(regular_savings).sum() *
                               REGULAR_SCORE;

                player_id++;
                string node_label = fmt::format("{}P{}", root_label, player_id);
            }

            // Critical victims are only successfully saved when rescued by
            // all players.
            rows = sampler->get_num_samples();
            cols = sampler->get_model()->get_time_steps();
            Eigen::MatrixXi critical_savings =
                Eigen::MatrixXi::Ones(rows, cols);

            while (sampler->get_model()->has_node_with_label(node_label)) {
                critical_savings =
                    critical_savings.array() *
                    (sampler->get_samples(node_label)(0, 0).array() ==
                     ASISTMultiPlayerMessageConverter::SAVING_CRITICAL)
                        .cast<int>();

                player_id++;
                string node_label = fmt::format("{}P{}", root_label, player_id);
            }

            final_score +=
                this->get_transitions(critical_savings).sum() * CRITICAL_SCORE;

            return {static_cast<double>(final_score)};
        }

        //----------------------------------------------------------------------
        // Getters & Setters
        //----------------------------------------------------------------------

    } // namespace model
} // namespace tomcat
