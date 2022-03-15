#pragma once

#include "utils/Definitions.h"

#include "pgm/inference/FactorGraph.h"
#include "pipeline/estimation/Estimator.h"

namespace tomcat {
    namespace model {

        /**
         * This estimator computes the estimates of a node by using
         * message-passing sum-product algorithm on top of a factor graph
         * originated from an unrolled DBN.
         */
        class SumProductEstimator : public Estimator {
          public:
            //------------------------------------------------------------------
            // Constructors & Destructor
            //------------------------------------------------------------------

            /**
             * Creates an instance of a sum-product estimator.
             *
             * @param model: DBN
             * @param inference_horizon: how many time steps in the future
             * estimations are going to be computed for
             * @param node_label: label of the node estimates are going to be
             * computed for
             * @param assignment: fixed assignment (for instance, estimates =
             * probability that the node assumes a value x, where x is the fixed
             * assignment). This parameter is optional when the inference
             * horizon is 0, but mandatory otherwise.
             * @param single_pass: whether a single pass forward and backward is
             * enough to estimate the marginals. In case it isn't, messages are
             * passed until stabilization (loopy belief algorithm).
             */
            SumProductEstimator(
                const std::shared_ptr<DynamicBayesNet>& model,
                int inference_horizon,
                const std::string& node_label,
                const Eigen::VectorXd& assignment = EMPTY_VECTOR,
                bool single_pass = true);

            ~SumProductEstimator();

            //------------------------------------------------------------------
            // Copy & Move constructors/assignments
            //------------------------------------------------------------------
            SumProductEstimator(const SumProductEstimator& estimator);

            SumProductEstimator&
            operator=(const SumProductEstimator& estimator);

            SumProductEstimator(SumProductEstimator&&) = default;

            SumProductEstimator& operator=(SumProductEstimator&&) = default;

            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------
            void prepare() override;

            void estimate(const EvidenceSet& new_data) override;

            void get_info(nlohmann::json& json) const override;

            std::string get_name() const override;

            //------------------------------------------------------------------
            // Getters & Setters
            //------------------------------------------------------------------

            void set_subgraph_window_size(int subgraph_window_size);

            void set_variable_window(bool variable_window);

          private:
            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            /**
             * Passes messages forward and backwards within a time step, and
             * then move forward taking the messages from boundaries nodes in
             * one time step to transition factors in the next time step.
             *
             * @param new_data: observed values for time steps not already
             * processed by the method
             */
            void estimate_forward_in_time(const EvidenceSet& new_data);

            /**
             * Computes messages from parents to child nodes in a given time
             * step.
             *
             * @param factor_graph: factor graph used to compute messages
             * @param time_step: time step to process
             * @param new_data: observed values for time steps not already
             * processed by the method
             *
             * @return whether any message changed
             */
            bool compute_forward_messages(const FactorGraph& factor_graph,
                                          int time_step,
                                          const EvidenceSet& new_data);

            /**
             * Computes messages from children to parent nodes in a given time
             * step.
             *
             * @param factor_graph: factor graph used to compute messages
             * @param time_step: time step to process
             * @param new_data: observed values for time steps not already
             * processed by the method
             * @param last_time_step: last time step to be processed
             *
             * @return whether any message changed
             */
            bool compute_backward_messages(const FactorGraph& factor_graph,
                                           int time_step,
                                           const EvidenceSet& new_data,
                                           int last_time_step);

            /**
             * Computes the a node's marginal for a given assignment in the
             * future, according to the the horizons provided to the estimator;
             *
             * @param time_step: current estimated time step. The predictions
             * are made beyond this time step up to the limit of the maximum
             * horizon provided to the estimator
             * @param num_data_points: number of data points in the data set
             * used for estimation
             *
             * @return Estimates for each data point in a single time step.
             */
            Eigen::MatrixXd get_predictions(int time_step, int num_data_points);

            /**
             * Appends the estimates matrix with a new column.
             *
             * @param new_column: new column with new estimates
             * @param index: index of the assignment for which the estimates
             * were computed. If there's a fixed assignment in the NodeEstimates
             * data, the index is always zero as there'll be estimates just for
             * a single assignment.
             */
            void add_column_to_estimates(const Eigen::VectorXd new_column,
                                         int index = 0);

            //------------------------------------------------------------------
            // Data members
            //------------------------------------------------------------------

            FactorGraph factor_graph;

            // Next time step to compute messages to the nodes in the factor
            // graph. Nodes with time steps before next_time_step already have
            // their messages computed.
            int next_time_step = 0;

            // Number of time slices up to the current time step being
            // processed, we update with message passing. By default, we only
            // pass messages to the current and the previous time step.
            int subgraph_window_size = 1;

            // If set to true, the subgraph_window_size will be adjusted
            // dynamically so that the full graph up to the current time step is
            // updated with message passing. That is, whatever is defined in the
            // attribute subgraph_window_size will be ignored.
            bool variable_subgraph_window = false;

            // Whether the marginals can be computed in a single
            // forward/backward pass or if they need to stabilize.
            bool single_pass;
        };

    } // namespace model
} // namespace tomcat
