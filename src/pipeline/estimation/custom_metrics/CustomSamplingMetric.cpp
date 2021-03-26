#include "CustomSamplingMetric.h"

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        CustomSamplingMetric::CustomSamplingMetric(int inference_horizon)
            : inference_horizon(inference_horizon) {}

        CustomSamplingMetric::~CustomSamplingMetric() {}

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        void
        CustomSamplingMetric::copy(const CustomSamplingMetric& custom_metric) {
            this->inference_horizon = custom_metric.inference_horizon;
        }

        Eigen::MatrixXi CustomSamplingMetric::get_transitions(
            const Eigen::MatrixXi& binary_matrix) const {
            Eigen::MatrixXi curr_values = binary_matrix.block(
                0, 0, binary_matrix.rows(), binary_matrix.cols() - 1);
            Eigen::MatrixXi next_values = binary_matrix.block(
                0, 1, binary_matrix.rows(), binary_matrix.cols() - 1);

            // Number of times 1 -> 0 is detected per row (mission sample)
            Eigen::MatrixXi successful_transitions =
                curr_values.array() * (1 - next_values.array());

            return successful_transitions;
        }

        //----------------------------------------------------------------------
        // Getters & Setters
        //----------------------------------------------------------------------
        int CustomSamplingMetric::get_inference_horizon() const {
            return inference_horizon;
        }

    } // namespace model
} // namespace tomcat
