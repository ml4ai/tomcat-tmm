#include "EstimationProcess.h"

#include "utils/EigenExtensions.h"

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        EstimationProcess::EstimationProcess(
            const AgentPtr& agent, const EstimateReporterPtr& reporter)
            : agent(agent), reporter(reporter) {}

        EstimationProcess::~EstimationProcess() {}

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        void EstimationProcess::set_training_data(
            const tomcat::model::EvidenceSet& training_data) {
            this->agent->set_training_data(training_data);
        }

        void EstimationProcess::keep_estimates() {
            this->agent->keep_estimates();
        }

        void EstimationProcess::clear_estimates() {
            this->agent->clear_estimates();
        }

        void EstimationProcess::prepare() {
            this->agent->prepare();
            this->last_time_step = -1;
        }

        void EstimationProcess::copy_estimation(
            const EstimationProcess& estimation) {
            this->agent = estimation.agent;
            this->display_estimates = estimation.display_estimates;
            this->last_time_step = estimation.last_time_step;
            this->reporter = estimation.reporter;
        }

        void EstimationProcess::estimate(const EvidenceSet& observations) {
            this->agent->estimate(observations);
            this->last_time_step += observations.get_time_steps();
            this->publish_last_estimates();
        }

        void EstimationProcess::get_info(nlohmann::json& json) const {
            if (this->display_estimates) {
                this->agent->get_info(json["agent"]);
            }
        }

        //----------------------------------------------------------------------
        // Getters & Setters
        //----------------------------------------------------------------------

        void EstimationProcess::set_display_estimates(bool display_estimates) {
            EstimationProcess::display_estimates = display_estimates;
        }

        const AgentPtr& EstimationProcess::get_agent() const { return agent; }

    } // namespace model
} // namespace tomcat
