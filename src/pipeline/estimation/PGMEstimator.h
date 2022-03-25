#pragma once

#include <memory>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

#include <eigen3/Eigen/Dense>
#include <nlohmann/json.hpp>

#include "pgm/DynamicBayesNet.h"
#include "pgm/EvidenceSet.h"
#include "pipeline/estimation/Estimator.h"
#include "utils/Definitions.h"
#include "utils/Tensor3.h"

namespace tomcat {
    namespace model {

        //------------------------------------------------------------------
        // Structs
        //------------------------------------------------------------------

        /**
         * This struct stores a node's label, assignment over which the
         * PGMEstimator must perform its computations and the estimates
         * calculated for that node.
         */
        struct NodeEstimates {

            std::string label;

            Eigen::VectorXd assignment;

            // Probabilities or densities calculated for n data points over
            // several time steps. If an assignment is provided, there will be
            // only one matrix in the vector containing the estimates for each
            // one of the data points and time steps. If no assignment is given,
            // there will be as many matrix estimates as the cardinality of the
            // node. In sum, there will be estimates for each possible
            // assignment the node can have.
            std::vector<Eigen::MatrixXd> estimates;

            // Any extra computation that needs to be reported
            std::vector<Eigen::MatrixXd> custom_data;
        };

        /**
         * This struct stores a node's label, assignment over which the
         * PGMEstimator must perform its computations and a list of the multiple
         * times estimates were calculated by the PGMEstimator.
         */
        struct CumulativeNodeEstimates {

            std::string label;

            Eigen::VectorXd assignment;

            // The external vector represents the content for each one of the
            // executions of the estimation process. In a cross-validation
            // procedure,the size of this vector will be defined by the number
            // of folds. The internal vector store estimates for each one of the
            // possible node's assignments. This will only happen if no fixed
            // assignment was provided, otherwise, this vector will have size 1
            // as it will contain estimated for a single assignment only. Single
            // assignments make sense when a inference horizon of size > 0 is
            // used.
            std::vector<std::vector<Eigen::MatrixXd>> estimates;

            std::vector<std::vector<Eigen::MatrixXd>> custom_data;
        };

        /**
         * Represents a generic PGMEstimator for a DBN model.
         */
        class PGMEstimator : public Estimator,
                             public std::enable_shared_from_this<PGMEstimator> {
          public:
            //------------------------------------------------------------------
            // Constructors & Destructor
            //------------------------------------------------------------------

            /**
             * Creates an empty PGMEstimator.
             */
            PGMEstimator() = default;

            /**
             * Creates an abstract PGMEstimator.
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
             */
            PGMEstimator(const std::shared_ptr<DynamicBayesNet>& model,
                         int inference_horizon,
                         const std::string& node_label,
                         const Eigen::VectorXd& assignment = EMPTY_VECTOR);

            /**
             * Creates an abstract compound PGMEstimator.
             *
             * @param model: DBN
             */
            explicit PGMEstimator(
                const std::shared_ptr<DynamicBayesNet>& model);

            virtual ~PGMEstimator() = default;

            //------------------------------------------------------------------
            // Copy & Move constructors/assignments
            //------------------------------------------------------------------

            // Copy constructor and assignment should be deleted to avoid
            // implicit slicing and loss of polymorphic behaviour in the
            // subclasses. To deep copy, the clone method must be used.
            PGMEstimator(const PGMEstimator&) = delete;

            PGMEstimator& operator=(const PGMEstimator&) = delete;

            PGMEstimator(PGMEstimator&&) = default;

            PGMEstimator& operator=(PGMEstimator&&) = default;

            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            /**
             * Returns estimates at a given time step.
             *
             * @param time_step: Time step to get the estimates from
             *
             * @return Estimates.
             */
            NodeEstimates get_estimates_at(int time_step) const;

            //------------------------------------------------------------------
            // Virtual functions
            //------------------------------------------------------------------

            /**
             * Clear last estimates and cumulative estimates computed by the
             * PGMEstimator.
             */
            void cleanup() override;

            /**
             * Initializations before the computation of estimates.
             */
            void prepare() override;

            /**
             * Accumulate estimates.
             */
            void keep_estimates() override;

            void get_info(nlohmann::json& json_estimators) const override;

            /**
             * Whether a node is being estimated by this PGMEstimator.
             *
             * @param node_label: Node's label
             *
             * @return
             */
            virtual bool
            is_computing_estimates_for(const std::string& node_label) const;

            /**
             * Retrieves the subset of PGMEstimator of an PGMEstimator. Compound
             * PGMEstimators can have multiple base PGMEstimators.
             *
             * @return Base PGMEstimators
             */
            virtual std::vector<std::shared_ptr<PGMEstimator>>
            get_base_estimators();

            /**
             * Whether prediction cases are binary. This is used to
             * differentiate cases in which it's mandatory to provide an
             * assignment to estimate probabilities in a horizon. Custom
             * PGMEstimators might change the default behavior, which is true,
             * if the use a positive horizon but computes estimates for multiple
             * assignments.
             *
             * @return
             */
            virtual bool is_binary_on_prediction() const;

            //------------------------------------------------------------------
            // Getters & Setters
            //------------------------------------------------------------------
            NodeEstimates get_estimates() const;

            CumulativeNodeEstimates get_cumulative_estimates() const;

            int get_inference_horizon() const;

            bool is_compound() const;

          protected:
            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            /**
             * Copies data members from another PGMEstimator.
             */
            void copy(const PGMEstimator& estimator);

            //------------------------------------------------------------------
            // Data members
            //------------------------------------------------------------------

            // Observed data to perform estimations. More data points can be
            // appended as estimations are made. Each derived class must store
            // computations to avoid recalculations as new data is available.
            // EvidenceSet test_data;

            // Node to compute estimates, its fixed assignment (optional if
            // inference_horizon = 0) and estimates
            NodeEstimates estimates;

            // Node to compute estimates, its fixed assignment and cumulative
            // estimates over several executions (if cross validation is used)
            CumulativeNodeEstimates cumulative_estimates;

            // An inference horizon determines if the task is a prediction (> 0)
            // or an inference (= 0). If it's a prediction, the horizon
            // determines up to how much further in the future predictions are
            // made.
            int inference_horizon = 0;

            // A compound PGMEstimator has a collection of concrete child
            // PGMEstimators. Compound PGMEstimators are used for performing a
            // computation that can be used by these concrete PGMEstimators.
            // Compound PGMEstimators cannot be used for evaluation as they do
            // not calculate an estimate for an specific node.
            bool compound;
        };

    } // namespace model
} // namespace tomcat
