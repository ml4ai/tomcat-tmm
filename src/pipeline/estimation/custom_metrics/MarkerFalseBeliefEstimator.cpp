#include "MarkerFalseBeliefEstimator.h"

#include "converter/ASISTMultiPlayerMessageConverter.h"
#include "utils/EigenExtensions.h"

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        MarkerFalseBeliefEstimator::MarkerFalseBeliefEstimator(
            const std::shared_ptr<DynamicBayesNet>& model) {
            this->model = model;
            this->inference_horizon = 0;
            this->estimates.label = LABEL;
            this->estimates.estimates =
                vector<Eigen::MatrixXd>(2); // stays in the hallway or enters room
        }

        MarkerFalseBeliefEstimator::~MarkerFalseBeliefEstimator() {}

        //----------------------------------------------------------------------
        // Copy & Move constructors/assignments
        //----------------------------------------------------------------------
        MarkerFalseBeliefEstimator::MarkerFalseBeliefEstimator(
            const MarkerFalseBeliefEstimator& estimator) {
            SamplerEstimator::copy(estimator);
        }

        MarkerFalseBeliefEstimator& MarkerFalseBeliefEstimator::operator=(
            const MarkerFalseBeliefEstimator& estimator) {
            SamplerEstimator::copy(estimator);
            return *this;
        }

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------

        void MarkerFalseBeliefEstimator::prepare() {
            this->estimates.estimates =
                vector<Eigen::MatrixXd>(2);
        }

        string MarkerFalseBeliefEstimator::get_name() const {
            return "marker_false_belief";
        }

        void MarkerFalseBeliefEstimator::estimate(
            const EvidenceSet& particles,
            const EvidenceSet& projected_particles,
            const EvidenceSet& marginals,
            int data_point_idx,
            int time_step) {


        }

        //----------------------------------------------------------------------
        // Getters & Setters
        //----------------------------------------------------------------------

    } // namespace model
} // namespace tomcat
