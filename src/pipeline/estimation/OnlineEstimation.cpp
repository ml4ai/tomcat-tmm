#include "OnlineEstimation.h"

#include <algorithm>
#include <sstream>
#include <thread>
#include <unordered_map>

#include <nlohmann/json.hpp>

#include "utils/EigenExtensions.h"

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        OnlineEstimation::OnlineEstimation(
            const AgentPtr& agent,
            const MessageBrokerConfiguration& config,
            const MsgConverterPtr& message_converter,
            const EstimateReporterPtr& reporter)
            : EstimationProcess(agent, reporter), config(config),
              message_converter(message_converter) {}

        OnlineEstimation::~OnlineEstimation() {}

        //----------------------------------------------------------------------
        // Copy & Move constructors/assignments
        //----------------------------------------------------------------------
        OnlineEstimation::OnlineEstimation(const OnlineEstimation& estimation)
            : EstimationProcess(estimation.agent, estimation.reporter) {
            this->copy_estimation(estimation);
        }

        OnlineEstimation&
        OnlineEstimation::operator=(const OnlineEstimation& estimation) {
            this->copy_estimation(estimation);
            return *this;
        }

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        void OnlineEstimation::prepare() {
            EstimationProcess::prepare();
            this->messages_to_process.clear();
            this->last_time_step = -1;
            this->evidence_metadata.clear();
        }

        void
        OnlineEstimation::copy_estimation(const OnlineEstimation& estimation) {
            EstimationProcess::copy_estimation(estimation);
            Mosquitto::copy_wrapper(estimation);
            this->config = estimation.config;
            this->message_converter = estimation.message_converter;
        }

        void OnlineEstimation::estimate(const EvidenceSet& test_data) {
            this->set_max_seconds_without_messages(this->config.timeout);
            this->connect(this->config.address,
                          this->config.port,
                          60,
                          this->config.num_connection_trials,
                          this->config.milliseconds_before_retrial);
            for (const string& topic :
                 this->message_converter->get_used_topics()) {
                this->subscribe(topic);
            }
            cout << "Waiting for mission to start..." << endl;
            thread estimation_thread(&OnlineEstimation::run_estimation_thread,
                                     this);
            this->loop();
            this->close();
            // Join because even if messages are not coming anymore, pending
            // data from previous messages could still be in the queue to
            // be processed.
            estimation_thread.join();
        }

        void OnlineEstimation::run_estimation_thread() {
            while (this->running || !this->messages_to_process.empty()) {
                EvidenceSet new_data =
                    this->get_next_data_from_pending_messages();
                if (!new_data.empty()) {
                    if (this->last_time_step < 0) {
                        cout << "Agent " << this->agent->get_id()
                             << " is awake and working..." << endl;
                    }

                    this->agent->estimate(new_data);
                    this->last_time_step++;
                    this->publish_last_estimates();

                    if (this->message_converter->is_mission_finished()) {
                        if (this->config.log_topic != "" && this->reporter) {
                            stringstream ss;
                            ss << "The maximum time step defined for the "
                                  "mission has been reached. Waiting for a new "
                                  "mission to start...";
                            string message =
                                this->reporter
                                    ->build_log_message(this->agent, ss.str())
                                    .dump();
                            this->publish(this->config.log_topic, message);
                        }

                        cout << "Waiting for a new mission to start..." << endl;
                        this->prepare();
                    }
                }
            }
        }

        EvidenceSet OnlineEstimation::get_next_data_from_pending_messages() {
            EvidenceSet new_data;

            while (!this->messages_to_process.empty() && new_data.empty()) {
                nlohmann::json message = this->messages_to_process.front();
                this->messages_to_process.pop();
                new_data.hstack(this->message_converter->get_data_from_message(
                    message, this->evidence_metadata));
                new_data.set_metadata(this->evidence_metadata);
            }

            return new_data;
        }

        void OnlineEstimation::publish_last_estimates() {
            auto messages = this->reporter->estimates_to_message(
                this->agent, this->last_time_step);
            for (const auto& message : messages) {
                this->publish(this->config.estimates_topic, message.dump());
            }
        }

        void OnlineEstimation::on_error(const string& error_message) {
            this->close();
            throw TomcatModelException(error_message);
        }

        void OnlineEstimation::on_message(const string& topic,
                                          const string& message) {
            nlohmann::json json_message = nlohmann::json::parse(message);
            json_message["topic"] = topic;
            this->messages_to_process.push(json_message);
        }

        void OnlineEstimation::on_time_out() {
            if (this->config.log_topic != "" && this->reporter) {
                stringstream ss;
                ss << "Connection time out!";
                string message =
                    this->reporter->build_log_message(this->agent, ss.str())
                        .dump();
                this->publish(this->config.log_topic, ss.str());
            }
        }

        void OnlineEstimation::get_info(nlohmann::json& json) const {
            EstimationProcess::get_info(json);
            json["process"] = "online";
        }

    } // namespace model
} // namespace tomcat
