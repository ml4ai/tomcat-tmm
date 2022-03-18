#pragma once

#include "pipeline/evaluation/Measure.h"
#include "utils/Definitions.h"

namespace tomcat {
    namespace model {

        /**
         * Class responsible for computing the F1 Score of the estimates
         * calculated for a given model by some estimator.
         */
        class F1Score : public Measure {
          public:
            inline static const std::string MACRO_NAME = "f1_macro";
            inline static const std::string MICRO_NAME = "f1_micro";

            //------------------------------------------------------------------
            // Constructors & Destructor
            //------------------------------------------------------------------

            /**
             * Creates an F1 Score measure.
             *
             * @param estimator: estimator used to compute the estimates
             * @param threshold: Probability threshold for predicting or
             * inferring the occurrence of an assignment as true
             * @param frequency_type: frequency at which estimates must be
             * computed
             * @param macro: whether macro or micro computation must be used for
             * the multi-class scenario.
             */
            F1Score(const std::shared_ptr<PGMEstimator>& estimator,
                    double threshold = 0.5,
                    FREQUENCY_TYPE frequency_type = all,
                    bool macro = true);

            ~F1Score();

            //------------------------------------------------------------------
            // Copy & Move constructors/assignments
            //------------------------------------------------------------------
            F1Score(const F1Score& f1_score);

            F1Score& operator=(const F1Score& f1_score);

            F1Score(F1Score&&) = default;

            F1Score& operator=(F1Score&&) = default;

            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------
            NodeEvaluation
            evaluate(const EvidenceSet& test_data) const override;

            void get_info(nlohmann::json& json) const override;

          private:
            //------------------------------------------------------------------
            // Data members
            //------------------------------------------------------------------

            /**
             * Computes F1 score for a given confusion matrix.
             *
             * @param confusion_matrix: confusion matrix
             *
             * @return F1 score
             */
            double get_score(const Eigen::MatrixXi& confusion_matrix) const;

            bool macro;
        };

    } // namespace model
} // namespace tomcat
