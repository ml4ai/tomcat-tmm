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
#include "utils/Definitions.h"
#include "utils/Tensor3.h"

namespace tomcat {
    namespace model {

        //------------------------------------------------------------------
        // Structs
        //------------------------------------------------------------------

        /**
         * This struct stores a node's label, assignment over which the
         * estimator must perform its computations and the estimates calculated
         * for that node.
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
        };

        /**
         * This struct stores a node's label, assignment over which the
         * estimator must perform its computations and a list of the multiple
         * times estimates were calculated by the estimator.
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
        };

        /**
         * Represents a generic estimator for a DBN model.
         */
        class Estimator : public std::enable_shared_from_this<Estimator> {
          public:
            //------------------------------------------------------------------
            // Constructors & Destructor
            //------------------------------------------------------------------

            /**
             * Creates an empty estimator.
             */
            Estimator();

            /**
             * Creates an abstract estimator.
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
            Estimator(const std::shared_ptr<DynamicBayesNet>& model,
                      int inference_horizon,
                      const std::string& node_label,
                      const Eigen::VectorXd& assignment = EMPTY_VECTOR);

            /**
             * Creates an abstract compound estimator.
             *
             * @param model: DBN
             */
            Estimator(const std::shared_ptr<DynamicBayesNet>& model);

            virtual ~Estimator();

            //------------------------------------------------------------------
            // Copy & Move constructors/assignments
            //------------------------------------------------------------------

            // Copy constructor and assignment should be deleted to avoid
            // implicit slicing and loss of polymorphic behaviour in the
            // subclasses. To deep copy, the clone method must be used.
            Estimator(const Estimator&) = delete;

            Estimator& operator=(const Estimator&) = delete;

            Estimator(Estimator&&) = default;

            Estimator& operator=(Estimator&&) = default;

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
             * estimator.
             */
            virtual void cleanup();

            /**
             * Initializations before the computation of estimates.
             *
             */
            virtual void prepare();

            /**
             * Store the last estimates computed in a list of cumulative
             * estimates.
             */
            virtual void keep_estimates();

            /**
             * Whether a node is being estimated by this estimator.
             *
             * @param node_label: Node's label
             *
             * @return
             */
            virtual bool
            is_computing_estimates_for(const std::string& node_label) const;

            /**
             * Retrieves the subset of estimator of an estimator. Compound
             * estimators can have multiple base estimators.
             *
             * @return Base estimators
             */
            virtual std::vector<std::shared_ptr<Estimator>>
            get_base_estimators();

            //------------------------------------------------------------------
            // Pure virtual functions
            //------------------------------------------------------------------

            /**
             * Computes new estimates for the new data. New data consists of
             * observed values for time steps after the last processed one.
             *
             * @param new_data: observed values for time steps not already
             * seen by the estimator
             */
            virtual void estimate(const EvidenceSet& new_data) = 0;

            /**
             * Writes information about the estimator in a json object.
             *
             * @param json: json object
             */
            virtual void get_info(nlohmann::json& json) const = 0;

            /**
             * Returns the name of the estimator.
             *
             * @return Estimator's name.
             */
            virtual std::string get_name() const = 0;

            //------------------------------------------------------------------
            // Getters & Setters
            //------------------------------------------------------------------
            NodeEstimates get_estimates() const;

            CumulativeNodeEstimates get_cumulative_estimates() const;

            int get_inference_horizon() const;

            void set_training_data(const EvidenceSet& training_data);

            const std::shared_ptr<DynamicBayesNet>& get_model() const;

            void set_show_progress(bool show_progress);

            bool is_compound() const;

          protected:
            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            /**
             * Copies data members from another estimator.
             */
            void copy_estimator(const Estimator& estimator);

            //------------------------------------------------------------------
            // Data members
            //------------------------------------------------------------------
            std::shared_ptr<DynamicBayesNet> model;

            // Data used to train the model. Baseline methods can use the
            // information in the training set to compute their estimations
            // instead of test data.
            EvidenceSet training_data;

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

            // Whether a progress bar must be shown as the estimations are
            // happening
            bool show_progress = true;

            // A compound estimator has a collection of concrete child
            // estimators. Compound estimators are used for performing a
            // computation that can be used by these concrete estimators.
            // Compound estimators cannot be used for evaluation as they do not
            // calculate an estimate for an specific node.
            bool compound;
        };

    } // namespace model
} // namespace tomcat
