#include "OnlineEstimation.h"

#include <algorithm>
#include <sstream>
#include <thread>

#include "fmt/format.h"
#include <nlohmann/json.hpp>

#include "converter/MessageConverter.h"

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
            const EstimateReporterPtr& reporter,
            const OnlineLoggerPtr& logger)
            : EstimationProcess(agent, reporter), config(config),
              message_converter(message_converter), logger(logger) {

            // This callback function will be invoked every time the converter
            // parses messages that need immediate responses.
            message_converter->set_callback_function(bind(
                &OnlineEstimation::on_request, this, std::placeholders::_1));
            this->prepare();
        }

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
            this->message_converter->start_new_mission();
            this->reporter->prepare();
        }

        void
        OnlineEstimation::copy_estimation(const OnlineEstimation& estimation) {
            EstimationProcess::copy_estimation(estimation);
            Mosquitto::copy_wrapper(estimation);
            this->config = estimation.config;
            this->message_converter = estimation.message_converter;
            this->logger = estimation.logger;
        }

        void OnlineEstimation::estimate(const EvidenceSet& test_data) {
            this->set_max_seconds_without_messages(this->config.timeout);
            if (this->logger) {
                this->logger->log(fmt::format("Trying to connect to {}:{}...",
                                              this->config.address,
                                              this->config.port));
            }
            this->connect(this->config.address,
                          this->config.port,
                          60,
                          this->config.num_connection_trials,
                          this->config.milliseconds_before_retrial);
            for (const string& topic :
                 this->message_converter->get_used_topics()) {
                this->subscribe(topic);
            }
            string log_msg = "Waiting for mission to start...";
            cout << log_msg << endl;
            if (this->logger) {
                this->logger->log("Connection established!");
                this->logger->log(log_msg);
            }
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
                        stringstream log_msg;
                        log_msg << "Agent " << this->agent->get_id()
                                << " is awake and working...";
                        cout << log_msg.str() << endl;
                        if (this->logger) {
                            this->logger->log(log_msg.str());
                            this->logger->log_first_evidence_set(new_data);
                        }
                    }
                    this->agent->estimate(new_data);
                    this->last_time_step++;
                    this->publish_last_estimates();
                    if (this->message_converter->is_mission_finished()) {
                        string log_msg =
                            "Waiting for a new mission to start...";
                        cout << log_msg << endl;
                        if (this->logger) {
                            this->logger->log(">>>MISSION END<<<");
                            this->logger->log(log_msg);
                        }
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
                nlohmann::json json_metadatas = nlohmann::json::array();
                json_metadatas.push_back(this->evidence_metadata);
                new_data.set_metadata(json_metadatas);
            }

            return new_data;
        }

        void OnlineEstimation::publish_last_estimates() {
            auto messages = this->reporter->translate_estimates_to_messages(
                this->agent, this->last_time_step);
            for (const auto& message : messages) {
                this->publish(this->config.intervention_topic, message.dump());
            }
        }

        void
        OnlineEstimation::on_request(const nlohmann::json& request_message) {
            auto message = this->reporter
                               ->build_message_by_request(this->agent,
                                                          request_message,
                                                          this->last_time_step)
                               .dump();
            string topic = this->reporter->get_request_response_topic(
                request_message, this->config);
            if (!message.empty() && !topic.empty()) {
                this->publish(topic, message);
            }
        }

        void OnlineEstimation::on_error(const string& error_message) {
            if (this->logger) {
                this->logger->log(error_message);
            }
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
            string log_msg = "Connection time out!";
            cout << log_msg << endl;
            if (this->logger) {
                this->logger->log(log_msg);
            }
        }

        void OnlineEstimation::get_info(nlohmann::json& json) const {
            EstimationProcess::get_info(json);
            json["process"] = "online";
        }

    } // namespace model
} // namespace tomcat
