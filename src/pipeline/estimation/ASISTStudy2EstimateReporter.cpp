#include "ASISTStudy2EstimateReporter.h"

#include "converter/ASISTMultiPlayerMessageConverter.h"

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        ASISTStudy2EstimateReporter::ASISTStudy2EstimateReporter() {}

        ASISTStudy2EstimateReporter::~ASISTStudy2EstimateReporter() {}

        //----------------------------------------------------------------------
        // Copy & Move constructors/assignments
        //----------------------------------------------------------------------
        ASISTStudy2EstimateReporter::ASISTStudy2EstimateReporter(
            const ASISTStudy2EstimateReporter& reporter) {}

        ASISTStudy2EstimateReporter& ASISTStudy2EstimateReporter::operator=(
            const ASISTStudy2EstimateReporter& reporter) {
            return *this;
        }

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        nlohmann::json ASISTStudy2EstimateReporter::get_header_section(
            const AgentPtr& agent) const {
            nlohmann::json header;
            header["timestamp"] = this->get_current_timestamp();
            header["message_type"] = "agent";
            header["version"] = "1.0";

            return header;
        }

        nlohmann::json
        ASISTStudy2EstimateReporter::get_msg_section(const AgentPtr& agent,
                                                     int data_point) const {
            nlohmann::json msg;
            msg["trial_id"] =
                agent->get_evidence_metadata()[data_point]["trial_id"];
            msg["experiment_id"] =
                agent->get_evidence_metadata()[data_point]["experiment_id"];
            msg["timestamp"] = this->get_current_timestamp();
            msg["source"] = agent->get_id();
            msg["sub_type"] = "prediction:state";
            msg["version"] = "1.0";

            return msg;
        }

        nlohmann::json ASISTStudy2EstimateReporter::get_data_section(
            const AgentPtr& agent, int time_step, int data_point) const {
            nlohmann::json data;
            const string& initial_timestamp =
                agent->get_evidence_metadata()[data_point]["initial_timestamp"];
            int elapsed_time =
                time_step *
                (int)agent->get_evidence_metadata()[data_point]["step_size"];

            data["created"] =
                this->get_elapsed_timestamp(initial_timestamp, elapsed_time);

            data["predictions"] = nlohmann::json::array();

            for (const auto& estimator : agent->get_estimators()) {
                for (const auto& base_estimator :
                     estimator->get_base_estimators()) {
                    int i = 1;
                    for (const string& player_name :
                         agent
                             ->get_evidence_metadata()[data_point]["players"]) {
                        if (base_estimator->get_estimates().label ==
                            "BlockKnowledgeP" + to_string(i)) {

                            nlohmann::json prediction;
                            prediction["unique_id"] = "Generate unique id";
                            prediction["subject"] = player_name;
                            // prediction["start"] = null; Estimates apply
                            // immediately
                            prediction["duration"] =
                                (int)agent->get_evidence_metadata()[data_point]
                                ["step_size"];
                            prediction["predicted_property"] = "Marker Legend";
                            prediction["probability_type"] = "float";

                            double prob_a =
                                base_estimator->get_estimates().estimates[0](
                                    data_point, time_step);
                            double prob_b =
                                base_estimator->get_estimates().estimates[1](
                                    data_point, time_step);
                            if (prob_a > prob_b) {
                                prediction["prediction"] = "VERSION A";
                                prediction["probability"] = prob_a;
                            }
                            else if (prob_a < prob_b) {
                                prediction["prediction"] = "VERSION B";
                                prediction["probability"] = prob_b;
                            }
                            else {
                                prediction["prediction"] = "UNDECIDED";
                                prediction["probability"] = 0.5;
                            }

                            prediction["confidence_type"] = "N.A.";
                            prediction["confidence"] = "N.A.";
                            prediction["explanation"] = "N.A.";

                            data["predictions"].push_back(prediction);
                        }
                        i++;
                    }
                }
            }

            return data;
        }

    } // namespace model
} // namespace tomcat
