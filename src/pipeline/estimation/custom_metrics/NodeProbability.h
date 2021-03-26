#pragma once

#include "pipeline/estimation/custom_metrics/CustomSamplingMetric.h"

namespace tomcat {
    namespace model {

        /**
         * Represents a metric to compute the probability of a given
         * node assignment (or all possible assignments) in a given horizon
         * of inference.
         */
        class NodeProbability : public CustomSamplingMetric {
          public:
            //------------------------------------------------------------------
            // Constructors & Destructor
            //------------------------------------------------------------------

            /**
             * Creates an instance of the node probability metric.
             *
             * @param node_label: label of the node which probability will be
             * computed for
             * @param assignment: assignment which probability will be
             * computed for
             * @param inference_horizon: window in the where estimates have to
             * be computed at.
             *
             */
            NodeProbability(std::string node_label,
                            Eigen::VectorXd assignment,
                            int inference_horizon);

            ~NodeProbability();

            //------------------------------------------------------------------
            // Copy & Move constructors/assignments
            //------------------------------------------------------------------

            NodeProbability(const NodeProbability& node_probability);

            NodeProbability& operator=(const NodeProbability& node_probability);

            NodeProbability(NodeProbability&&) = default;

            NodeProbability& operator=(NodeProbability&&) = default;

            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            /**
             * Use the samples generated for a node in the inference horizon
             * to estimate the probability of observing an assignment or all
             * possible assignments of the node in a fixed window.
             *
             * @param sampler: sampler that holds the generated samples
             * @param time_step: time step at which the metric is being
             * computed
             *
             * @return Single probability for the node in the window
             * (time_step : time_step + horizon].
             */
            std::vector<double>
            calculate(const std::shared_ptr<Sampler>& sampler,
                      int time_step) const override;

            //------------------------------------------------------------------
            // Getters & Setters
            //------------------------------------------------------------------
            const std::string& get_node_label() const;

            const Eigen::VectorXd& get_assignment() const;

          private:
            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            /**
             * Copy attributes from another instance.
             *
             * @param node_probability: another instance
             */
            void copy(const NodeProbability& node_probability);

            /**
             * Computes the frequency of samples generated within a time
             * window with values between a range.
             *
             * @param samples: samples generated within a window of interest
             * @param low: lower bound of the range
             * @param high: upper bound of the range
             *
             * @return frequency of samples in the range
             */
            double get_probability_in_range(const Eigen::MatrixXd& samples,
                                            double low,
                                            double high) const;

            //------------------------------------------------------------------
            // Data members
            //------------------------------------------------------------------
            std::string node_label;

            Eigen::VectorXd assignment;
        };

    } // namespace model
} // namespace tomcat
