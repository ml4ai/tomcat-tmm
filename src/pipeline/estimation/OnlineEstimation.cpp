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
            const MessageBrokerConfiguration& config,
            const std::shared_ptr<Agent>& agent)
            : config(config), agent(agent) {}

        OnlineEstimation::~OnlineEstimation() {}

        //----------------------------------------------------------------------
        // Copy & Move constructors/assignments
        //----------------------------------------------------------------------
        OnlineEstimation::OnlineEstimation(const OnlineEstimation& estimation) {
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
            this->time_step = 0;
        }

        void
        OnlineEstimation::copy_estimation(const OnlineEstimation& estimation) {
            EstimationProcess::copy_estimation(estimation);
            Mosquitto::copy_wrapper(estimation);
            this->config = estimation.config;
            this->agent = estimation.agent;
        }

        void OnlineEstimation::estimate(const EvidenceSet& test_data) {
            this->set_max_seconds_without_messages(this->config.timeout);
            this->connect(this->config.address,
                          this->config.port,
                          60,
                          this->config.num_connection_trials,
                          this->config.milliseconds_before_retrial);
            for (const string& topic : this->agent->get_topics_to_subscribe()) {
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
                    for (auto estimator : this->estimators) {
                        EstimationProcess::estimate(estimator, new_data);
                    }
                    this->publish_last_estimates();

                    if (this->time_step == 0) {
                        cout << "Agent " << this->agent->get_id()
                             << " is awake \\o/ and working..." << endl;
                    }

                    if (this->agent->is_mission_finished()) {
                        const string& topic = this->agent->get_log_topic();
                        if (topic != "") {
                            stringstream ss;
                            ss << "The maximum time step defined for the "
                                  "mission has been reached. Waiting for a new "
                                  "mission to start...";
                            string message =
                                this->agent->build_log_message(ss.str()).dump();
                            this->publish(topic, message);
                        }
                        cout << "Waiting for a new mission to start..." << endl;
                        this->prepare();
                    }
                    else {
                        this->time_step++;
                    }
                }
            }
        }

        EvidenceSet OnlineEstimation::get_next_data_from_pending_messages() {
            EvidenceSet new_data;

            while (!this->messages_to_process.empty() && new_data.empty()) {
                nlohmann::json message = this->messages_to_process.front();
                this->messages_to_process.pop();
                new_data.hstack(this->agent->message_to_data(message));
            }

            return new_data;
        }

        void OnlineEstimation::publish_last_estimates() {
            vector<nlohmann::json> messages = this->agent->estimates_to_message(
                this->estimators, this->time_step);
            const string& topic = this->agent->get_estimates_topic();

            for(const auto& message : messages) {
                this->publish(topic, message.dump());
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
            const string& topic = this->agent->get_log_topic();
            if (topic != "") {
                stringstream ss;
                ss << "Connection time out!";
                string message =
                    this->agent->build_log_message(ss.str()).dump();
                this->publish(topic, ss.str());
            }
        }

        void OnlineEstimation::get_info(nlohmann::json& json) const {
            EstimationProcess::get_info(json);
            json["process"] = "online";
        }

    } // namespace model
} // namespace tomcat
