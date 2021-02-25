#pragma once

#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <nlohmann/json.hpp>

#include "pgm/EvidenceSet.h"
#include "utils/Definitions.h"

namespace tomcat {
    namespace model {

        /**
         * Generic class for converting messages related to the observations
         * from Minecraft to a matrix format that can be read by a DBN.
         */
        class MessageConverter {
          public:
            //------------------------------------------------------------------
            // Constructors & Destructor
            //------------------------------------------------------------------

            /**
             * Creates an empty abstract message converter.
             */
            MessageConverter();

            /**
             * Creates an abstract instance of the message converter.
             *
             * @param num_seconds: total number of seconds in a mission
             * @param time_step_size: size of a time step (in seconds) between
             * observations
             */
            MessageConverter(int num_seconds, int time_step_size);

            virtual ~MessageConverter();

            //------------------------------------------------------------------
            // Copy & Move constructors/assignments
            //------------------------------------------------------------------

            // Copy constructor and assignment should be deleted to avoid
            // implicit slicing and loss of polymorphic behaviour in the
            // subclasses. To deep copy, the clone method must be used.
            MessageConverter(const MessageConverter&) = delete;

            MessageConverter& operator=(const MessageConverter&) = delete;

            MessageConverter(MessageConverter&&) = default;

            MessageConverter& operator=(MessageConverter&&) = default;

            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            /**
            * Converts messages from files in a given folder to files for each
            * observable node, consisting of tensors with the observations for
            * each mission sample and time step in the mission.
            *
            * @param messages_dir: directory where the message files are
            * @param data_dir: directory where data must be saved
            */
            void convert_messages(const std::string& messages_dir,
                                  const std::string& data_dir);

            //------------------------------------------------------------------
            // Pure virtual functions
            //------------------------------------------------------------------

            /**
             * Parses a json message object and, if related to an observation of
             * a given node, returns it values. If it's a world observation,
             * updates the time step. Otherwise, ignore the message.
             *
             * @param message: json object as a string
             *
             * @return List of pairs containing a nodes' labels and observations
             * for the current time step.
             */
            virtual EvidenceSet
            get_data_from_message(const nlohmann::json& json_message,
                                  nlohmann::json& json_mission_log) = 0;

          protected:
            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            /**
             * Copies the data members from another converter instance.
             *
             * @param converter: converter to copy the data members from
             */
            void copy_converter(const MessageConverter& converter);

            //------------------------------------------------------------------
            // Pure virtual functions
            //------------------------------------------------------------------

            virtual std::map<std::string, nlohmann::json>
            filter(const std::string& messages_filepath) const = 0;

            //------------------------------------------------------------------
            // Data members
            //------------------------------------------------------------------

            // Number of seconds between observations
            int time_step_size;

            // Number of time steps in a mission
            int time_steps;

            // Indicates whether the messages should be processed as for a
            // new mission. This is used by the online conversion to
            // recognize when to restart processing messages.
            bool new_mission;

          private:
            /**
             * Checks if the directory where the data will be saved has a
             * conversion log file and, if it does, use it to avoid
             * reprocessing messages that were previously converted.
             *
             * @param messages_dir: directory where the message files are
             * located
             * @param data_dir: directory where the converted messages
             * will be saved
             *
             * @return List of message files that were not previously
             * processed.
             */
            std::unordered_set<std::string>
            get_unprocessed_message_filenames(const std::string& messages_dir,
                                              const std::string& data_dir);
        };

    } // namespace model
} // namespace tomcat
