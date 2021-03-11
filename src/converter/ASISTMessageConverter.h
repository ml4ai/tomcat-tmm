#pragma once

#include <iostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <boost/filesystem.hpp>
#include <eigen3/Eigen/Dense>
#include <nlohmann/json.hpp>

#include "MessageConverter.h"

#include "utils/Definitions.h"
#include "utils/Tensor3.h"

namespace tomcat {
    namespace model {

        /**
         * Converts messages from the ASIST testbed to a format that the model
         * can process.
         */
        class ASISTMessageConverter : public MessageConverter {
          public:
            //------------------------------------------------------------------
            // Constructors & Destructor
            //------------------------------------------------------------------

            /**
             * Creates an instance of the message converter.
             *
             * @param num_seconds: number of seconds of a mission
             * @param time_step_size: seconds between observations
             */
            ASISTMessageConverter(int num_seconds, int time_step_size);

            virtual ~ASISTMessageConverter();

            //------------------------------------------------------------------
            // Copy & Move constructors/assignments
            //------------------------------------------------------------------
            ASISTMessageConverter(const ASISTMessageConverter&) = delete;

            ASISTMessageConverter&
            operator=(const ASISTMessageConverter&) = delete;

            ASISTMessageConverter(ASISTMessageConverter&&) = default;

            ASISTMessageConverter& operator=(ASISTMessageConverter&&) = default;

            //------------------------------------------------------------------
            // Pure virtual functions
            //------------------------------------------------------------------

            /**
             * Gets message topics used to data extraction.
             *
             * @return Message topics.
             */
            virtual std::unordered_set<std::string> get_used_topics() const = 0;

            //------------------------------------------------------------------
            // Getters & Setters
            //------------------------------------------------------------------
            std::time_t get_mission_initial_timestamp() const;

            int get_mission_trial_number() const;

            const std::string& get_experiment_id() const;

          protected:
            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            std::map<std::string, nlohmann::json>
            filter(const std::string& messages_filepath) const override;

            /**
             * Copies attributes from another converter.
             *
             * @param converter: original converter
             */
            void copy_converter(const ASISTMessageConverter& converter);

            /**
             * Converts a string with remaining minutes and seconds to the total
             * number of seconds elapsed since the mission started.
             *
             * @param time: string containing the remaining time formatted as
             * mm : ss
             *
             * @return Elapsed time in seconds.
             */
            int get_elapsed_time(const std::string& time);

            //------------------------------------------------------------------
            // Data members
            //------------------------------------------------------------------
            time_t mission_initial_timestamp;

            int mission_trial_number = -1;

            std::string experiment_id;
        };

    } // namespace model
} // namespace tomcat
