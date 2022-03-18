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
        Agent::Agent(const string& id, const string& version)
            : id(id), version(version) {}

        Agent::~Agent() {}

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
                for (const auto& estimator : this->estimators) {
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

        void Agent::show_progress(bool show_progress) {
            for (auto& estimator : this->estimators) {
                estimator->set_show_progress(show_progress);
            }
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
                estimator->get_info(json["estimators"]);
            }
        }

        //----------------------------------------------------------------------
        // Getters & Setters
        //----------------------------------------------------------------------
        const string& Agent::get_id() const { return id; }

        const string& Agent::get_version() const { return version; }

        void Agent::set_ignored_observations(
            const unordered_set<std::string>& ignored_observations) {
            this->ignored_observations = ignored_observations;
        }

        const EstimatorPtrVec& Agent::get_estimators() const {
            return estimators;
        }

        const nlohmann::json& Agent::get_evidence_metadata() const {
            return evidence_metadata;
        }
    } // namespace model
} // namespace tomcat
