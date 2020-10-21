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

namespace tomcat {
    namespace model {

        /**
         * Converts messages from the TA3 testbed to a format that the model can
         * process.
         */
        class TA3MessageConverter : public MessageConverter {
          public:
            //------------------------------------------------------------------
            // Types, Enums & Constants
            //------------------------------------------------------------------
            // Total number of time steps in a mission.
            inline static const int T = 600;

            // Node labels
            inline static const std::string ROOM = "Room";
            inline static const std::string SG = "Green";
            inline static const std::string SY = "Yellow";
            inline static const std::string Q = "TrainingCondition";
            inline static const std::string BEEP = "Beep";

            inline static const std::string METADATA_FILENAME = "metadata.json";

            //------------------------------------------------------------------
            // Constructors & Destructor
            //------------------------------------------------------------------

            /**
             * Creates an instance of the TA3 message converter.
             *
             * @param map_config_filepath: path of the map configuration file
             * @param time_gap: gap (in seconds) between observations.
             */
            TA3MessageConverter(const std::string& map_config_filepath,
                                int time_gap = 1);

            ~TA3MessageConverter();

            //------------------------------------------------------------------
            // Copy & Move constructors/assignments
            //------------------------------------------------------------------
            TA3MessageConverter(const TA3MessageConverter& converter);

            TA3MessageConverter&
            operator=(const TA3MessageConverter& converter);

            TA3MessageConverter(TA3MessageConverter&&) = default;

            TA3MessageConverter& operator=(TA3MessageConverter&&) = default;

            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            void convert_offline(const std::string& input_dir,
                                 const std::string& output_dir) override;

            std::unordered_map<std::string, double>
            convert_online(const nlohmann::json& message) override;

          private:
            //------------------------------------------------------------------
            // Static functions
            //------------------------------------------------------------------

            /**
             * Reads messages from a file and returns them sorted by timestamp.
             *
             * @param filepath: path of the file where the messages are stored.
             *
             * @return Sorted messages
             */
            static std::vector<nlohmann::json>
            get_sorted_messages_in(const std::string& filepath);

            /**
             * Converts a string with remaining minutes and seconds to the total
             * number of seconds for the mission to end.
             *
             * @param time: string containing the remaining time formatted (mm :
             * ss)
             *
             * @return Remaining time in seconds.
             */
            int get_remaining_seconds_from(const std::string& time);

            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            /**
             * Fills the map of observations with default not observed values.
             */
            void init_observations();

            /**
             * Returns the content of a metadata file in the output directory as
             * a json object or creates a new one in case there's no metadata
             * file in such directory.
             *
             * @param metadata_dir: directory where the metadata file is
             *
             * @return Json object with metadata of previous conversions.
             */
            nlohmann::json get_metadata(const std::string& metadata_dir);

            /**
             * Returns the set of message file names that were previously
             * processed.
             *
             * @param json_metadata: json object containing the metadata of
             * previous conversions attempts
             * @param all: if true, returns all message filenames processed,
             * regardless if the conversion was successful or not. Otherwise,
             * only returns filenames of messages that could be converted
             * successfully.
             *
             * @return Set of message file names.
             */
            std::unordered_set<std::string>
            get_processed_message_filenames(const nlohmann::json& json_metadata,
                                            bool all = true);

            /**
             * Returns filepath of message files not yet converted.
             *
             * @param messages_dir: directory where the message files are
             * @param processed_message_filenames: set of message filenames that
             * were previously processed for conversion
             *
             * @return Paths of message files never processed.
             */
            std::vector<boost::filesystem::path>
            get_message_filepaths(const std::string& messages_dir,
                                  const std::unordered_set<std::string>&
                                      processed_message_filenames);

            /**
             * Loads map of area configuration as a hash map to easily determine
             * if an area is a room or not by its id.
             *
             * @param map_config_filepath: path of the map configuration file
             */
            void
            load_map_area_configuration(const std::string& map_config_filepath);

            /**
             * Returns observation related to victim saving.
             *
             * @param json_message: json message containing a victim saving
             * event.
             *
             * @return Victim saving event observation.
             */
            void
            fill_victim_saving_observation(const nlohmann::json& json_message);

            /**
             * Returns observation related to being in a room or not.
             *
             * @param json_message: json message containing information about
             * the room the player is in.
             *
             * @return Room observation.
             */
            void fill_room_observation(const nlohmann::json& json_message);

            /**
             * Returns observation about the beep event.
             *
             * @param json_message: json message containing information about
             * the beep played.
             *
             * @return Observation regarding the beep playing.
             */
            void fill_beep_observation(const nlohmann::json& json_message);

            /**
             * Saves the contents of a metadata json object to a metadata file.
             *
             * @param json_metadata: json object that contains the metadata
             * information
             * @param output_dir: directory where the metadata must be saved
             */
            void save_metadata(const nlohmann::json& json_metadata,
                               const std::string& output_dir);

            /**
             * Merges observations with previously converted messages and saves
             * the full data matrix for each node in the observations map.
             *
             * @param observations_per_node: matrix of observations per node
             * @param output_dir: directory where the observations must be saved
             */
            void merge_and_save_observations(
                const std::unordered_map<std::string, Eigen::MatrixXd>&
                    observations_per_node,
                const std::string& output_dir);

            //------------------------------------------------------------------
            // Data members
            //------------------------------------------------------------------

            // Indicates whether a message informing about the mission start was
            // received. Messages received before the mission starts will be
            // ignored.
            bool mission_started = false;

            // Stores the id of all possible areas in the map along with a flag
            // indicating whether the area is a room or not (e.g, yard, hallway
            // etc.).
            std::unordered_map<std::string, bool> map_area_configuration;

            // Last observations per node.
            std::unordered_map<std::string, double> last_observations_per_node;

            int training_condition = NO_OBS;

            // Timestamp when the mission starts.
            std::string initial_timestamp;
        };

    } // namespace model
} // namespace tomcat
