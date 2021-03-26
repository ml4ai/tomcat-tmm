#pragma once

#include <memory>
#include <thread>

#include <nlohmann/json.hpp>

#include "EstimationProcess.h"

#include "utils/Definitions.h"
#include "utils/Mosquitto.h"
#include "utils/SynchronizedQueue.h"
#include "pipeline/estimation/Agent.h"

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
            // Structs
            //------------------------------------------------------------------

            /**
             * This struct contains information needed to connect to a message
             * broker to either subscribe or publish to a topic.
             */
            struct MessageBrokerConfiguration {
               int timeout = 9999;

                std::string address;
                int port;
                int num_connection_trials;
                int milliseconds_before_retrial;
            };

            //------------------------------------------------------------------
            // Constructors & Destructor
            //------------------------------------------------------------------

            /**
             * Creates an online estimation process.
             *
             * @param estimator: type of estimation to be performed
             * @param agent: agent who talks to the message bus
             */
            OnlineEstimation(
                const MessageBrokerConfiguration& config,
                const std::shared_ptr<Agent>& agent);

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
             * Returns next set of observations from the pending messages in the
             * queue.
             *
             * @return Evidence set.
             */
            EvidenceSet get_next_data_from_pending_messages();

            /**
             * Publishes last computed estimates to the message bus.
             */
            void publish_last_estimates();

            //------------------------------------------------------------------
            // Data members
            //------------------------------------------------------------------
            MessageBrokerConfiguration config;

            std::shared_ptr<Agent> agent;

            // Number of time steps the estimation already processed.
            int time_step;

            // Messages received from the message bus and stored to be processed
            // by the estimation threads.
            SynchronizedQueue<nlohmann::json> messages_to_process;
        };

    } // namespace model
} // namespace tomcat
