#include "OfflineEstimation.h"

#include <thread>

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        OfflineEstimation::OfflineEstimation(const AgentPtr& agent)
            : EstimationProcess(agent, nullptr) {}

        OfflineEstimation::OfflineEstimation(
            const AgentPtr& agent,
            const EstimateReporterPtr& reporter,
            const std::string& report_filepath)
            : EstimationProcess(agent, reporter) {

            if (reporter && !report_filepath.empty()) {
                this->report_file.open(report_filepath, ios_base::app);
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
                // In the offline estimation, estimates for all time steps
                // and data points will be processed at the end of the
                // estimation function. Therefore, we need to process time
                // by time to publish estimates over time.

                // The reporter can process all time steps at once
                auto messages = this->reporter->translate_estimates_to_messages(
                    this->agent, NO_OBS);

                if (this->report_file.is_open()) {
                    for (const auto& message : messages) {
                        this->report_file << message << "\n";
                    }
                    this->report_file.close();
                }
                else {
                    for (const auto& message : messages) {
                        cout << message << "\n";
                    }
                }
            }
        }

    } // namespace model
} // namespace tomcat
