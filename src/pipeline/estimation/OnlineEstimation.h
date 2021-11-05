#pragma once

#include <memory>
#include <thread>
#include <unordered_set>
#include <functional>

#include <nlohmann/json.hpp>

#include "EstimationProcess.h"

#include "pipeline/estimation/Agent.h"
#include "utils/Definitions.h"
#include "utils/Mosquitto.h"
#include "utils/SynchronizedQueue.h"
#include "utils/OnlineConfig.h"

namespace tomcat {
    namespace model {

        /**
         * Class responsible for computing estimates for a model in an online
         * fashion. It listens to a message bus topic to compute estimates in
         * real time as data is observed.
         */
        class OnlineEstimation : public EstimationProcess, public Mosquitto {
          public:
            //------------------------------------------------------------------
            // Constructors & Destructor
            //------------------------------------------------------------------

            /**
             * Creates an online estimation process.
             *
             * @param agent: agent used in the estimation
             * @param config: the message broker configuration
             * @param message_converter: classes responsible to translate json
             * messages to observations
             * @param reporter: class responsible for reporting estimates
             * computed during the process
             */
            OnlineEstimation(const AgentPtr& agent,
                             const MessageBrokerConfiguration& config,
                             const MsgConverterPtr& message_converter,
                             const EstimateReporterPtr& reporter);

            ~OnlineEstimation();

            //------------------------------------------------------------------
            // Copy & Move constructors/assignments
            //------------------------------------------------------------------
            OnlineEstimation(const OnlineEstimation& estimation);

            OnlineEstimation& operator=(const OnlineEstimation& estimation);

            // The synchronized queue used in this class has no move
            // constructor, so let's not allow this class to be moved as well..
            OnlineEstimation(OnlineEstimation&&) = delete;

            OnlineEstimation& operator=(OnlineEstimation&&) = delete;

            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------
            void estimate(const EvidenceSet& test_data) override;

          protected:
            void prepare() override;

            void on_error(const std::string& error_message) override;

            void on_message(const std::string& topic,
                            const std::string& message) override;

            void on_time_out() override;

            void get_info(nlohmann::json& json) const override;

            /**
             * Publishes last computed estimates to the message bus.
             */
            void publish_last_estimates() override;

          private:
            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            /**
             * Copies data members from another online estimation process.
             */
            void copy_estimation(const OnlineEstimation& estimation);

            /**
             * Function executed by a thread responsible for calculating the
             * estimates for a single estimator.
             *
             * @param estimator: estimator
             * @param test_data: data to estimate values over
             */
            void run_estimation_thread();

            /**
             * Publishes a heartbeat message to let the system know that the
             * agent is alive
             */
            void publish_heartbeat();

            /**
             * Publishes a message as soon as a mission starts.
             */
            void publish_start_of_mission_message();

            /**
             * Publishes a message as soon as a mission ends.
             */
            void publish_end_of_mission_message();

            /**
             * Returns next set of observations from the pending messages in the
             * queue.
             *
             * @return Evidence set.
             */
            EvidenceSet get_next_data_from_pending_messages();

            /**
             * Callback function executed by the message converter's request.
             */
            void on_request(const nlohmann::json& request_message);

            //------------------------------------------------------------------
            // Data members
            //------------------------------------------------------------------
            MessageBrokerConfiguration config;

            // Messages received from the message bus and stored to be processed
            // by the estimation threads.
            SynchronizedQueue<nlohmann::json> messages_to_process;

            MsgConverterPtr message_converter;

            // Information about the trial being processed
            nlohmann::json evidence_metadata;
        };

    } // namespace model
} // namespace tomcat
