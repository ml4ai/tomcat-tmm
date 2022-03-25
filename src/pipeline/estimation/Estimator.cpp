#include "Estimator.h"

#include <fmt/format.h>
#include <sstream>

#include "asist/study2/FinalTeamScoreEstimator.h"
#include "asist/study2/IndependentMapVersionAssignmentEstimator.h"
#include "asist/study2/MapVersionAssignmentEstimator.h"
#include "asist/study2/NextAreaOnNearbyMarkerEstimator.h"
#include "asist/study3/ASISTStudy3InterventionEstimator.h"

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        Estimator::Estimator(const shared_ptr<Model>& model) : model(model) {}

        //------------------------------------------------------------------
        // Static functions
        //------------------------------------------------------------------

        EstimatorPtr
        Estimator::factory(const std::string& estimator_name,
                           const ModelPtr& model,
                           const nlohmann::json& json_settings,
                           FREQUENCY_TYPE frequency_type,
                           const std::unordered_set<int>& fixed_time_steps,
                           int num_jobs) {

            const unordered_set<string> pgm_estimators = {
                FinalTeamScoreEstimator::NAME,
                MapVersionAssignmentEstimator::NAME,
                IndependentMapVersionAssignmentEstimator::NAME,
                NextAreaOnNearbyMarkerEstimator::NAME};

            EstimatorPtr estimator;
            if (EXISTS(estimator_name, pgm_estimators)) {
                if (const auto& dbn =
                        dynamic_pointer_cast<DynamicBayesNet>(model)) {

                    if (estimator_name == FinalTeamScoreEstimator::NAME) {
                        estimator = make_shared<FinalTeamScoreEstimator>(
                            dbn, frequency_type);
                    }
                    else if (estimator_name ==
                             MapVersionAssignmentEstimator::NAME) {
                        estimator = make_shared<MapVersionAssignmentEstimator>(
                            dbn, frequency_type);
                    }
                    else if (estimator_name ==
                             IndependentMapVersionAssignmentEstimator::NAME) {
                        estimator = make_shared<
                            IndependentMapVersionAssignmentEstimator>(
                            dbn, frequency_type);
                    }
                    else if (estimator_name ==
                             NextAreaOnNearbyMarkerEstimator::NAME) {
                        estimator =
                            make_shared<NextAreaOnNearbyMarkerEstimator>(
                                dbn, json_settings);
                    }
                }
                else {
                    throw TomcatModelException(fmt::format(
                        "Estimator {} is only defined for DBN models.",
                        estimator_name));
                }
            }
            else {
                if (estimator_name == ASISTStudy3InterventionEstimator::NAME) {
                    estimator =
                        make_shared<ASISTStudy3InterventionEstimator>(model);
                }
            }

            return estimator;
        }

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        void Estimator::copy(const Estimator& estimator) {
            this->model = estimator.model;
            this->training_data = estimator.training_data;
            this->show_progress = estimator.show_progress;
        }

        void Estimator::prepare() {}

        void Estimator::cleanup() {}

        void Estimator::keep_estimates() {}

        void Estimator::set_show_progress(bool show_progress) {
            this->show_progress = show_progress;
        }

        //----------------------------------------------------------------------
        // Getters & Setters
        //----------------------------------------------------------------------

        void Estimator::set_training_data(const EvidenceSet& training_data) {
            this->training_data = training_data;
        }

        const shared_ptr<Model>& Estimator::get_model() const { return model; }

    } // namespace model
} // namespace tomcat
