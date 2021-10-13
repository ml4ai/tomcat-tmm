#pragma once

#include <iostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <boost/filesystem.hpp>
#include <eigen3/Eigen/Dense>
#include <nlohmann/json.hpp>

#include "ASISTMessageConverter.h"

#include "utils/Definitions.h"
#include "utils/Tensor3.h"

namespace tomcat {
    namespace model {

        /**
         * Converts Study 3 messages from the ASIST testbed.
         */
        class ASISTStudy3MessageConverter : public ASISTMessageConverter {
          public:
            // Node Labels
            inline static const std::string ELAPSED_SECOND = "ElapsedSecond";
            inline static const std::string TEAM_SCORE = "TeamScore";

            //------------------------------------------------------------------
            // Constructors & Destructor
            //------------------------------------------------------------------

            /**
             * Creates an instance of the message converter.
             *
             * @param num_seconds: number of seconds of a mission
             * @param time_step_size: seconds between observations
             * @param map_filepath: path of the map configuration json file
             * @param num_players: number of players in the mission
             */
            ASISTStudy3MessageConverter(int num_seconds,
                                        int time_step_size,
                                        const std::string& map_filepath,
                                        int num_players);

            ~ASISTStudy3MessageConverter();

            //------------------------------------------------------------------
            // Copy & Move constructors/assignments
            //------------------------------------------------------------------
            ASISTStudy3MessageConverter(
                const ASISTStudy3MessageConverter& converter);

            ASISTStudy3MessageConverter&
            operator=(const ASISTStudy3MessageConverter& converter);

            ASISTStudy3MessageConverter(ASISTStudy3MessageConverter&&) =
                default;

            ASISTStudy3MessageConverter&
            operator=(ASISTStudy3MessageConverter&&) = default;

            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------
            bool is_valid_message_file(
                const boost::filesystem::directory_entry& file) const override;

            std::unordered_set<std::string> get_used_topics() const override;

          protected:
            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            /**
             * Copies attributes from another converter.
             *
             * @param converter: original converter
             */
            void copy_converter(const ASISTStudy3MessageConverter& converter);

            EvidenceSet parse_before_mission_start(
                const nlohmann::json& json_message,
                nlohmann::json& json_mission_log) override;

            EvidenceSet parse_after_mission_start(
                const nlohmann::json& json_message,
                nlohmann::json& json_mission_log) override;

            void fill_observation(const nlohmann::json& json_message) override;

            void prepare_for_new_mission() override;

            void do_offline_conversion_extra_validations() const override;

            void parse_individual_message(
                const nlohmann::json& json_message) override;

          private:
            struct Player {
                std::string id;
                std::string callsign; // red, blue or green
                std::string unique_id;
                std::string name;
            };

            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            /**
             * Loads map of area configuration as a hash map to easily determine
             * if an area is a room or not by its id.
             *
             * @param map_filepath: path of the map configuration json file
             */
            void load_map_area_configuration(const std::string& map_filepath);

            /**
             * Adds a player to the list of players in the mission.
             *
             * @param player: player
             */
            void add_player(const Player& player);

            /**
             * Extracts the trial number from a string containing the trial
             * number as a suffix.
             *
             * @param textual_trial_number: trial number in the format:
             * T000<number>
             *
             * @return Numeric trial number
             */
            int get_numeric_trial_number(
                const std::string& textual_trial_number) const;

            /**
             * Fills map info and marker legend for each one of the
             * participants in the trial.
             *
             * @param json_client_info: json object containing information
             * about the kind of map and marker legend received per player
             * @param json_log: json containing extra information about data
             * conversion
             *
             */
            void fill_players(const nlohmann::json& json_client_info,
                              nlohmann::json& json_log);

            /**
             * Gets the observations accumulated so far and creates an evidence
             * set with them.
             *
             * @return Evidence set.
             */
            EvidenceSet build_evidence_set_from_observations();

            /**
             * Writes any remaining information to the log at the end of a
             * mission.
             *
             * @param json_log: json containing extra information about data
             * conversion
             */
            void
            write_to_log_on_mission_finished(nlohmann::json& json_log) const;

            /**
             * Stores the current scoreboard.
             *
             * @param json_message
             */
            void parse_scoreboard_message(const nlohmann::json& json_message);

            /**
             * Stores live player info
             *
             * @param json_message: message
             * @param player_number: player number
             */
            void parse_player_state_message(const nlohmann::json& json_message,
                                            int player_number);

            //------------------------------------------------------------------
            // Data members
            //------------------------------------------------------------------

            int num_players;
            int next_time_step;

            // Numbers are sequential numbers starting from zero and indicate
            // the position in the vector of observations.
            std::unordered_map<std::string, int> player_id_to_number;
            std::vector<Player> players;

            // Evidence
            Tensor3 team_score;
        };

    } // namespace model
} // namespace tomcat
