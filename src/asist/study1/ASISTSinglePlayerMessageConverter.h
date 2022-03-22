#pragma once

#include <iostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <boost/filesystem.hpp>
#include <eigen3/Eigen/Dense>
#include <nlohmann/json.hpp>

#include "asist/ASISTMessageConverter.h"

#include "utils/Definitions.h"
#include "utils/Tensor3.h"

namespace tomcat {
    namespace model {

        /**
         * Converts messages from the ASIST testbed for single-player
         * missions to a format that the model can process.
         */
        class ASISTSinglePlayerMessageConverter : public ASISTMessageConverter {
          public:
            // Area values
            const static int HALLWAY = 0;
            const static int ROOM = 1;
            const static int ROOM_ENTRANCE = 2;

            // Task values
            const static int NO_TASK = 0;
            const static int SAVING_REGULAR = 1;
            const static int SAVING_CRITICAL = 2;

            //------------------------------------------------------------------
            // Constructors & Destructor
            //------------------------------------------------------------------

            /**
             * Creates an instance of the message converter.
             *
             * @param num_seconds: number of seconds of a mission
             * @param time_step_size: seconds between observations
             * @param map_filepath: path of the map configuration json file
             */
            ASISTSinglePlayerMessageConverter(int num_seconds,
                                              int time_step_size,
                                              const std::string& map_filepath);

            ~ASISTSinglePlayerMessageConverter();

            //------------------------------------------------------------------
            // Copy & Move constructors/assignments
            //------------------------------------------------------------------
            ASISTSinglePlayerMessageConverter(
                const ASISTSinglePlayerMessageConverter& converter);

            ASISTSinglePlayerMessageConverter&
            operator=(const ASISTSinglePlayerMessageConverter& converter);

            ASISTSinglePlayerMessageConverter(
                ASISTSinglePlayerMessageConverter&&) = default;

            ASISTSinglePlayerMessageConverter&
            operator=(ASISTSinglePlayerMessageConverter&&) = default;

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
            void
            copy_converter(const ASISTSinglePlayerMessageConverter& converter);

            EvidenceSet parse_before_mission_start(
                const nlohmann::json& json_message,
                nlohmann::json& json_mission_log) override;

            EvidenceSet parse_after_mission_start(
                const nlohmann::json& json_message,
                nlohmann::json& json_mission_log) override;

            void fill_observation(const nlohmann::json& json_message) override;

            void prepare_for_new_mission() override;

          private:
            //------------------------------------------------------------------
            // Structs
            //------------------------------------------------------------------
            struct Point {
                double x;
                double y;

                Point(double x, double y) : x(x), y(y) {}
                ~Point() {}

                double distance(const Point& p) {
                    return sqrt(pow(p.x - this->x, 2) + pow(p.y - this->y, 2));
                }
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

            //------------------------------------------------------------------
            // Data members
            //------------------------------------------------------------------

            // Stores the id of all possible areas in the map along with a flag
            // indicating whether the area is a room or not (e.g, yard, hallway
            // etc.).
            std::unordered_map<std::string, bool> map_area_configuration;

            // Position at the center of each door
            std::vector<Point> door_positions;

            Tensor3 training_condition;
            Tensor3 difficulty;
            Tensor3 area;
            Tensor3 task;
            Tensor3 beep;
        };

    } // namespace model
} // namespace tomcat
