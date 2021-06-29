#include "Agent.h"

#include <boost/date_time/posix_time/posix_time.hpp>

#include "utils/EigenExtensions.h"

namespace tomcat {
    namespace model {

        using namespace std;
        namespace pt = boost::posix_time;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        Agent::Agent(const string& id,
                     const string& estimates_topic,
                     const string& log_topic)
            : id(id), estimates_topic(estimates_topic), log_topic(log_topic) {}

        Agent::~Agent() {}

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------

        void Agent::copy(const Agent& agent) {
            this->id = agent.id;
            this->estimates_topic = agent.estimates_topic;
            this->log_topic = agent.log_topic;
        }

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------

        void Agent::set_training_data(
            const tomcat::model::EvidenceSet& training_data) {
            for (auto& estimator : this->estimators) {
                estimator->set_training_data(training_data);
            }
        }

        void Agent::keep_estimates() {
            for (auto& estimator : this->estimators) {
                estimator->keep_estimates();
            }
        }

        void Agent::clear_estimates() {
            for (auto& estimator : this->estimators) {
                estimator->cleanup();
            }
        }

        void Agent::estimate(const EvidenceSet& observations) {
            EvidenceSet relevant_observations = observations;

            for (const auto& node_label : this->ignored_observations) {
                relevant_observations.remove(node_label);
            }

            if (!observations.empty()) {
                for (auto estimator : this->estimators) {
                    estimator->estimate(relevant_observations);
                }
            }

            // This number will be consistent ven across multiple calls to this
            // function during a estimation process.
            this->evidence_metadata = observations.get_metadata();
        }

        void Agent::add_estimator(const EstimatorPtr& estimator) {
            this->estimators.push_back(estimator);
        }

        void Agent::prepare() {
            for (auto& estimator : this->estimators) {
                estimator->prepare();
            }
        }

        void Agent::get_info(nlohmann::json& json) const {
            json["id"] = this->id;
            json["estimators"] = nlohmann::json::array();
            for (const auto& estimator : this->estimators) {
                // An estimator can be compound and comprised of multiple
                // base estimators, which are the ones that actually
                // store the estimates.
                for (const auto& base_estimator :
                     estimator->get_base_estimators()) {

                    nlohmann::json json_estimator;
                    base_estimator->get_info(json_estimator);

                    CumulativeNodeEstimates cumulative_estimates =
                        base_estimator->get_cumulative_estimates();

                    json_estimator["node_label"] = cumulative_estimates.label;
                    json_estimator["node_assignment"] =
                        to_string(cumulative_estimates.assignment);
                    json_estimator["executions"] = nlohmann::json::array();

                    for (const auto& estimates_matrix_per_execution :
                         cumulative_estimates.estimates) {

                        nlohmann::json json_execution;
                        json_execution["estimates"] = nlohmann::json::array();

                        for (const auto& estimates_matrix :
                             estimates_matrix_per_execution) {
                            json_execution["estimates"].push_back(
                                to_string(estimates_matrix));
                        }

                        json_estimator["executions"].push_back(json_execution);
                    }

                    json["estimators"].push_back(json_estimator);
                }
            }
        }

        vector<nlohmann::json>
        Agent::estimates_to_message(int time_step) const {
            // No specific format to be published
            return {};
        }

        nlohmann::json Agent::build_log_message(const string& log) const {
            nlohmann::json message;
            message["AgentID"] = this->id;
            message["log"] = log;

            return message;
        }

        //----------------------------------------------------------------------
        // Getters & Setters
        //----------------------------------------------------------------------
        const string& Agent::get_id() const { return id; }

        const string& Agent::get_estimates_topic() const {
            return estimates_topic;
        }

        const string& Agent::get_log_topic() const { return log_topic; }
    } // namespace model
} // namespace tomcat
