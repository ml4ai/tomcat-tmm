#include "OfflineEstimation.h"

#include <thread>

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        OfflineEstimation::OfflineEstimation() : EstimationProcess(nullptr) {}

        OfflineEstimation::OfflineEstimation(
            const EstimateReporterPtr& reporter,
            const std::string& report_filepath)
            : EstimationProcess(reporter) {

            if (report_filepath != "") {
                this->report_file.open(report_filepath);
            }
        }

        OfflineEstimation::~OfflineEstimation() {
            if (this->report_file.is_open()) {
                this->report_file.close();
            }
        }

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        void OfflineEstimation::get_info(nlohmann::json& json) const {
            EstimationProcess::get_info(json);
            json["process"] = "offline";
        }

        void OfflineEstimation::publish_last_estimates() {
            if (this->reporter) {
                for (auto& agent : this->agents) {
                    // In the offline estimation, estimates for all time steps
                    // and data points will be processed at the end of the
                    // estimation function. Therefore, we need to process time
                    // by time to publish estimates over time.
                    for (int t = 0; t <= this->last_time_step; t++) {
                        auto messages =
                            this->reporter->estimates_to_message(agent, t);

                        if (this->report_file.is_open()) {
                            for(const auto& message : messages) {
                                this->report_file << message << "\n";
                            }
                        }
                        else {
                            for(const auto& message : messages) {
                                cout << message << "\n";
                            }
                        }
                    }
                }
            }
        }

    } // namespace model
} // namespace tomcat
