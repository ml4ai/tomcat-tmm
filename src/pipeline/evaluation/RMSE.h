#pragma once

#include "pipeline/evaluation/Measure.h"
#include "utils/Definitions.h"

namespace tomcat {
    namespace model {

        /**
         * Class responsible for computing the RMSE of the estimates
         * calculated for a given model by some estimator.
         */
        class RMSE : public Measure {
          public:
            inline static const std::string NAME = "RMSE";

            //------------------------------------------------------------------
            // Constructors & Destructor
            //------------------------------------------------------------------

            /**
             * Creates a RMSE measure.
             *
             * @param estimator: estimator used to compute the estimates
             * @param use_last_estimate: whether only the estimates in the last
             * time step must be taken into consideration
             * @param frequency_type: frequency at which estimates must be
             * computed
             */
            RMSE(const std::shared_ptr<Estimator>& estimator,
                 FREQUENCY_TYPE frequency_type = all);

            ~RMSE();

            //------------------------------------------------------------------
            // Copy & Move constructors/assignments
            //------------------------------------------------------------------
            RMSE(const RMSE& rmse);

            RMSE& operator=(const RMSE& rmse);

            RMSE(RMSE&&) = default;

            RMSE& operator=(RMSE&&) = default;

            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------
            NodeEvaluation
            evaluate(const EvidenceSet& test_data) const override;

            void get_info(nlohmann::json& json) const override;

          private:
            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            /**
             * Computes RMSE score
             *
             * @param true_values: ground truth
             * @param true_values: estimates
             * @return
             */
            double get_score(const Eigen::MatrixXd& true_values,
                             const Eigen::MatrixXd& estimated_values) const;
        };

    } // namespace model
} // namespace tomcat
