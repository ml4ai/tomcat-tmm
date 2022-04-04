#pragma once

#include <memory>

#include "pgm/EvidenceSet.h"
#include "pipeline/estimation/Agent.h"
#include "reporter/EstimateReporter.h"
#include "utils/Definitions.h"

namespace tomcat {
    namespace model {

        /**
         * Logs information to a file during the online estimation process.
         */
        class OnlineLogger {
          public:
            //------------------------------------------------------------------
            // Constructors & Destructor
            //------------------------------------------------------------------

            /**
             * Creates a logger
             *
             * @param log_filepath: path of the file where information will be
             * logged to.
             *
             */
            OnlineLogger(const std::string& log_filepath);

            virtual ~OnlineLogger();

            //------------------------------------------------------------------
            // Copy & Move constructors/assignments
            //------------------------------------------------------------------

            OnlineLogger(const OnlineLogger&) = delete;

            OnlineLogger& operator=(const OnlineLogger&) = delete;

            OnlineLogger(OnlineLogger&&) = default;

            OnlineLogger& operator=(OnlineLogger&&) = default;

            //------------------------------------------------------------------
            // Virtual functions
            //------------------------------------------------------------------

            virtual void create_header();

            virtual void log(const std::string& text);

          protected:
            //------------------------------------------------------------------
            // Data members
            //------------------------------------------------------------------
            bool initialized = false;

            std::ofstream log_file;
        };

    } // namespace model
} // namespace tomcat
