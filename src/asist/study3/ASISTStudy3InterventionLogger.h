#pragma once

#include <memory>

#include "pgm/EvidenceSet.h"
#include "pipeline/estimation/Agent.h"
#include "pipeline/estimation/OnlineLogger.h"
#include "reporter/EstimateReporter.h"
#include "utils/Definitions.h"

namespace tomcat {
    namespace model {

        class ASISTStudy3InterventionLogger : public OnlineLogger {
          public:
            inline static const std::string NAME =
                "asist_study3_intervention_logger";

            //------------------------------------------------------------------
            // Constructors & Destructor
            //------------------------------------------------------------------

            ASISTStudy3InterventionLogger(const std::string& log_filepath);

            ~ASISTStudy3InterventionLogger() = default;

            //------------------------------------------------------------------
            // Copy & Move constructors/assignments
            //------------------------------------------------------------------

            ASISTStudy3InterventionLogger(
                const ASISTStudy3InterventionLogger&) = delete;

            ASISTStudy3InterventionLogger&
            operator=(const ASISTStudy3InterventionLogger&) = delete;

            ASISTStudy3InterventionLogger(ASISTStudy3InterventionLogger&&) =
                default;

            ASISTStudy3InterventionLogger&
            operator=(ASISTStudy3InterventionLogger&&) = default;

            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            void create_header() override;

            void log(const std::string& text) override;

            void log_first_evidence_set(const EvidenceSet& data) override;

            void log_watch_intervention(int time_step, const std::string& text);

            void log_cancel_intervention(int time_step,
                                         const std::string& text);

            void log_trigger_intervention(int time_step,
                                          const std::string& text);

            void log(int time_step, const std::string& text);

            //------------------------------------------------------------------
            // Getters & Setters
            //------------------------------------------------------------------

            void set_time_step_size(int time_step_size);
            void set_num_time_steps(int num_time_steps);

          private:
            //------------------------------------------------------------------
            // Static functions
            //------------------------------------------------------------------

            static std::string change_extension(const std::string& filepath);

            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            void log_common_info(const std::string& mission_timer,
                                 const std::string& text);

            std::string time_step_to_mission_timer(int time_step) const;

            //------------------------------------------------------------------
            // Data members
            //------------------------------------------------------------------

            int time_step_size = 1;
            int num_time_steps = 0;
        };

    } // namespace model
} // namespace tomcat
