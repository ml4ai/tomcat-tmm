#include "OnlineLogger.h"

#include <iomanip>

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        OnlineLogger::OnlineLogger(const string& log_filepath) {
            this->log_file.open(log_filepath, ios_base::app);
        }

        OnlineLogger::~OnlineLogger() {
            if (this->log_file.is_open()) {
                this->log_file << ">>>> END <<<<\n\n";
                this->log_file.close();
            }
        }

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        void OnlineLogger::create_header() {
            this->log_file << std::setw(30) << left << "Timestamp"
                           << " | ";
            this->log_file << std::setw(100) << left << "Text"
                           << " |";
            this->log_file << "\n";
        }

        void OnlineLogger::log(const string& text) {
            if (!this->initialized) {
                this->create_header();
                this->initialized = true;
            }

            this->log_file << std::setw(30) << left
                           << Timer::get_current_timestamp() << " | ";
            this->log_file << std::setw(100) << left << text << " |";
            this->log_file << "\n";
            this->log_file.flush();
        }

    } // namespace model
} // namespace tomcat
