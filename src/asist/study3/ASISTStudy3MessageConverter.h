#pragma once

#include <iostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <eigen3/Eigen/Dense>
#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include "asist/ASISTMessageConverter.h"
#include "utils/Definitions.h"
#include "utils/Tensor3.h"

namespace tomcat {
    namespace model {

        /**
         * Converts Study 3 messages from the ASIST testbed.
         */
        class ASISTStudy3MessageConverter : public ASISTMessageConverter {
          public:
            // Labels used in the variables collected
            struct Labels {
                inline const static std::string ENCOURAGEMENT = "Encouragement";
            };

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

            void
            fill_observation(const nlohmann::json& json_message,
                             nlohmann::json& json_mission_log) override;

            void prepare_for_new_mission() override;

            void do_offline_conversion_extra_validations() const override;

            void parse_individual_message(
                const nlohmann::json& json_message) override;

          private:
            struct Player {
              public:
                Player(const std::string& id, const std::string& color)
                    : id(id) {
                    this->color = color;
                    boost::algorithm::to_lower(this->color);

                    if (this->color == "red") {
                        this->index = 0;
                    }
                    else if (this->color == "green") {
                        this->index = 1;
                    }
                    else if (this->color == "blue") {
                        this->index = 2;
                    }
                    else {
                        throw TomcatModelException(fmt::format(
                            "Invalid player color {}.", this->color));
                    }
                }

                std::string id;
                std::string color;
                std::string role;
                int index;
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
            void parse_map(const std::string& map_filepath);

            /**
             * Fills map info and marker legend for each one of the
             * participants in the trial.
             *
             * @param json_client_info: json object containing information
             * about the kind of map and marker legend received per player
             */
            void parse_players(const nlohmann::json& json_client_info);

            /**
             * Update player's role
             *
             * @param json_message: json containing the role information
             * @param json_mission_log: metadata for the mission
             */
            void
            parse_role_selection_message(const nlohmann::json& json_message,
                                         nlohmann::json& json_mission_log);

            /**
             * Gets the observations accumulated so far and creates an evidence
             * set with them.
             *
             * @return Evidence set.
             */
            EvidenceSet collect_time_step_evidence();

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
             * @param json_message: json containing the scoreboard information
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

            /**
             * Find player object by its ID.
             *
             * @param player_id: player id
             *
             * @return
             */
            Player& get_player_by_id(const std::string& player_id);

            //------------------------------------------------------------------
            // Data members
            //------------------------------------------------------------------

            int num_players;
            int next_time_step;
            int num_players_with_role;

            std::unordered_map<std::string, int> player_id_to_index;
            std::vector<Player> players;

            // Data
            int num_encouragement_utterances;





        };

    } // namespace model
} // namespace tomcat
