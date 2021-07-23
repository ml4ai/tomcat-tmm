#pragma once

#include <memory>
#include <string>
#include <unordered_set>

#include <nlohmann/json.hpp>

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
         * estimator performed its computations and the evaluations calculated
         * for that node.
         */
        struct NodeEvaluation {

            std::string label;

            Eigen::VectorXd assignment;

            Eigen::MatrixXd evaluation;

            Eigen::MatrixXi confusion_matrix;
        };

        /**
         * Represents some measurement that can be performed over estimates.
         *
         */
        class Measure {
          public:
            //------------------------------------------------------------------
            // Types, Enums & Constants
            //------------------------------------------------------------------

            enum FREQUENCY_TYPE { all, last, fixed };

            //------------------------------------------------------------------
            // Constructors & Destructor
            //------------------------------------------------------------------

            /**
             * Creates an empty measure.
             *
             * @param threshold: Probability threshold for predicting or
             * inferring the occurrence of an assignment as true
             */
            Measure();

            /**
             * Creates an abstract measure.
             *
             * @param estimator: estimator used to compute the estimates
             * @param threshold: Probability threshold for predicting or
             * inferring the occurrence of an assignment as true
             * @param frequency_type: frequency at which estimates must be
             * computed
             */
            Measure(const std::shared_ptr<Estimator>& estimator,
                    double threshold = 0.5,
                    FREQUENCY_TYPE frequency_type = all);

            virtual ~Measure();

            //------------------------------------------------------------------
            // Copy & Move constructors/assignments
            //------------------------------------------------------------------
            Measure(const Measure&) = delete;

            Measure& operator=(const Measure&) = delete;

            Measure(Measure&&) = default;

            Measure& operator=(Measure&&) = default;

            //------------------------------------------------------------------
            // Pure virtual functions
            //------------------------------------------------------------------

            /**
             * Calculates the measure over a data set.
             *
             * @param test_data: data to calculate the measure over
             *
             * @return Evaluation for over the estimates computed by an
             * estimator.
             */
            virtual NodeEvaluation
            evaluate(const EvidenceSet& test_data) const = 0;

            /**
             * Writes information about the measure in a json object.
             *
             * @param json: json object
             */
            virtual void get_info(nlohmann::json& json) const = 0;

            //------------------------------------------------------------------
            // Getters & Setters
            //------------------------------------------------------------------

            void set_fixed_steps(const std::unordered_set<int>& fixed_steps);

          protected:
            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            /**
             * Copies the data members from another measure.
             *
             * @param measure: measure to copy data members from
             */
            void copy_measure(const Measure& measure);

            /**
             * Computes the confusion matrices between estimated values
             * previously computed for a model and real values. This assumes the
             * estimates were already computed by the estimator associated to
             * the measure. This also assumes that if an estimator has inference
             * horizon positive, it must have been given an fixed assignment so
             * that the problem becomes binary.
             *
             * If the frequency type of the evaluation is fixed, one matrix is
             * computed per fixed time step. Otherwise, only one matrix is
             * computed including all time steps or just the last one depending
             * on the frequency type assigned to the measure.
             *
             * @param test_data: data with true values to compare the
             * estimates against
             *
             * @return Confusion matrices per fixed time step.
             */
            std::vector<Eigen::MatrixXi>
            get_confusion_matrices(const EvidenceSet& test_data) const;

            //------------------------------------------------------------------
            // Data members
            //------------------------------------------------------------------
            // The estimates computed and stored in the estimator will be used
            // to evaluate the measure.
            std::shared_ptr<Estimator> estimator;

            // Probability threshold for predicting or inferring the occurrence
            // of an assignment as true
            double threshold = 0.5;

            FREQUENCY_TYPE frequency_type;
            std::unordered_set<int> fixed_steps;
        };

    } // namespace model
} // namespace tomcat
