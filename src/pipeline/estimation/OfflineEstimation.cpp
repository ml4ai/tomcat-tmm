#include "OfflineEstimation.h"

#include <thread>

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        OfflineEstimation::OfflineEstimation() {}

        OfflineEstimation::OfflineEstimation(std::ostream& output_stream)
            : output_stream(&output_stream) {}

        OfflineEstimation::~OfflineEstimation() {}

        //----------------------------------------------------------------------
        // Copy & Move constructors/assignments
        //----------------------------------------------------------------------
        OfflineEstimation::OfflineEstimation(
            const OfflineEstimation& estimation) {
            this->copy_estimation(estimation);
        }

        OfflineEstimation&
        OfflineEstimation::operator=(const OfflineEstimation& estimation) {
            this->copy_estimation(estimation);
            return *this;
        }

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------

        void OfflineEstimation::copy_estimation(
            const OfflineEstimation& estimation) {
            EstimationProcess::copy_estimation(estimation);
            this->output_stream = estimation.output_stream;
        }

        void OfflineEstimation::get_info(nlohmann::json& json) const {
            EstimationProcess::get_info(json);
            json["process"] = "offline";
        }

        void OfflineEstimation::publish_last_estimates() {
            if (this->output_stream) {
                vector<nlohmann::json> messages;
                for (auto& agent : this->agents) {
                    // In the offline estimation, estimates for all time steps
                    // and data points will be processed at the end of the
                    // estimation function. Therefore, we need to process time
                    // by time to publish estimates over time.
                    for (int t = 0; t < this->time_step; t++) {
                        (*this->output_stream)
                            << agent->estimates_to_message(t) << "\n";
                    }
                }
            }
        }

    } // namespace model
} // namespace tomcat
