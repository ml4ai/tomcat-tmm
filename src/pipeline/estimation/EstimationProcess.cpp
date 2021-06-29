#include "EstimationProcess.h"

#include "utils/EigenExtensions.h"

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        EstimationProcess::EstimationProcess() {}

        EstimationProcess::~EstimationProcess() {}

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        void EstimationProcess::set_training_data(
            const tomcat::model::EvidenceSet& training_data) {
            for (auto& agent : this->agents) {
                agent->set_training_data(training_data);
            }
        }

        void EstimationProcess::add_agent(const AgentPtr& agent) {
            this->agents.push_back(agent);
        }

        void EstimationProcess::keep_estimates() {
            for (auto& agent : this->agents) {
                agent->keep_estimates();
            }
        }

        void EstimationProcess::clear_estimates() {
            for (auto& agent : this->agents) {
                agent->clear_estimates();
            }
        }

        void EstimationProcess::prepare() {
            for (auto& agent : this->agents) {
                agent->prepare();
            }
        }

        void EstimationProcess::copy_estimation(
            const EstimationProcess& estimation) {
            this->agents = estimation.agents;
            this->display_estimates = estimation.display_estimates;
            this->time_step = estimation.time_step;
        }

        void EstimationProcess::estimate(const EvidenceSet& observations) {
            for (auto& agent : this->agents) {
                agent->estimate(observations);
            }
            this->time_step += observations.get_time_steps();
            this->publish_last_estimates();
        }

        void EstimationProcess::get_info(nlohmann::json& json) const {
            if (this->display_estimates) {
                json["agents"] = nlohmann::json::array();
                for (const auto& agent : this->agents) {
                    nlohmann::json json_agent;
                    agent->get_info(json_agent);
                    json["agents"].push_back(json_agent);
                }
            }
        }

        //----------------------------------------------------------------------
        // Getters & Setters
        //----------------------------------------------------------------------

        void EstimationProcess::set_display_estimates(bool display_estimates) {
            EstimationProcess::display_estimates = display_estimates;
        }

        const AgentPtrVec& EstimationProcess::get_agents() const {
            return agents;
        }

    } // namespace model
} // namespace tomcat
