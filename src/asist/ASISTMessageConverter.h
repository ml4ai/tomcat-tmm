#pragma once

#include <iostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <boost/filesystem.hpp>
#include <eigen3/Eigen/Dense>
#include <nlohmann/json.hpp>

#include "converter/MessageConverter.h"

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
            // Static functions
            //------------------------------------------------------------------

            /**
             * Checks if a message if from the type informed
             *
             * @param json_message: message
             * @param type: type
             *
             * @return
             */
            static bool is_message_of(const nlohmann::json& json_message,
                                      const std::string& type);

            /**
             * Checks if a message if from the type and sub-type informed
             *
             * @param json_message: message
             * @param type: type
             * @param sub_type: sub-type
             *
             * @return
             */
            static bool is_message_of(const nlohmann::json& json_message,
                                      const std::string& type,
                                      const std::string& sub_type);

            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            void convert_messages(const std::string& messages_dir,
                                  const std::string& data_dir) override;

            EvidenceSet
            get_data_from_message(const nlohmann::json& json_message,
                                  nlohmann::json& json_mission_log) override;

            //------------------------------------------------------------------
            // Getters & Setters
            //------------------------------------------------------------------
            std::time_t get_mission_initial_timestamp() const;

            int get_mission_trial_number() const;

            const std::string& get_experiment_id() const;

          protected:
            //------------------------------------------------------------------
            // Structs
            //------------------------------------------------------------------

            struct BoundingBox {
                int x1 = 0;
                int x2 = 0;
                int z1 = 0;
                int z2 = 0;

                BoundingBox() {}

                BoundingBox(int x1, int x2, int z1, int z2)
                    : x1(x1), x2(x2), z1(z1), z2(z2) {}

                std::vector<BoundingBox>
                get_horizontal_splits(int num_sections = 2) const {
                    std::vector<BoundingBox> sections(num_sections);

                    int height = (z2 - z1) / num_sections;
                    int initial_z = z1;
                    for (int i = 0; i < num_sections; i++) {
                        if (i == num_sections - 1) {
                            // If the heights cannot be evenly split, the last
                            // section will get a different height amount.
                            height = z2 - initial_z;
                        }
                        BoundingBox section(
                            x1, x2, initial_z, initial_z + height);
                        initial_z += height + 1;
                        sections[i] = section;
                    }

                    return sections;
                }

                std::vector<BoundingBox>
                get_vertical_splits(int num_sections = 2) const {
                    std::vector<BoundingBox> sections(num_sections);

                    int width = (x2 - x1) / num_sections;
                    int initial_x = x1;
                    for (int i = 0; i < num_sections; i++) {
                        if (i == num_sections - 1) {
                            // If the widths cannot be evenly split, the last
                            // section will get a different width amount.
                            width = x2 - initial_x;
                        }
                        BoundingBox section(
                            initial_x, initial_x + width, z1, z2);
                        initial_x += width + 1;
                        sections[i] = section;
                    }

                    return sections;
                }
            };

            struct Position {
                double x = 0;
                double z = 0;

                Position() {}

                Position(double x, double z) : x(x), z(z) {}

                bool is_inside(const BoundingBox& box) const {
                    return this->x <= box.x2 && this->x >= box.x1 &&
                           this->z <= box.z2 && this->z >= box.z1;
                }

                bool equals(const Position& other_position) const {
                    return this->x == other_position.x &&
                           this->z == other_position.z;
                }

                double get_distance(const Position& other_position) const {
                    return sqrt(pow(this->x - other_position.x, 2) +
                                pow(this->z - other_position.z, 2));
                }
            };

            //------------------------------------------------------------------
            // Pure virtual functions
            //------------------------------------------------------------------
            /**
             * Parse message before mission starts.
             *
             * @param json_message: json message.
             * @param json_mission_log: includes info to be put in the
             * conversion metadata file.
             *
             * @return Data collected from the parsed message.
             */
            virtual EvidenceSet
            parse_before_mission_start(const nlohmann::json& json_message,
                                       nlohmann::json& json_mission_log) = 0;

            /**
             * Parse message before mission starts.
             *
             * @param json_message: json message.
             * @param json_mission_log: includes info to be put in the
             * conversion metadata file.
             *
             * @return Data collected from the parsed message.
             */
            virtual EvidenceSet
            parse_after_mission_start(const nlohmann::json& json_message,
                                      nlohmann::json& json_mission_log) = 0;

            /**
             * Parse message and fill the appropriate observation tensor.
             *
             * @param json_message: json message with a particular observation.
             * @param json_message: json message to update mission metadata
             */
            virtual void fill_observation(const nlohmann::json& json_message,
                                          nlohmann::json& json_mission_log) = 0;

            /**
             * Clea nup before new mission starts
             */
            virtual void prepare_for_new_mission() = 0;

            //------------------------------------------------------------------
            // Virtual functions
            //------------------------------------------------------------------

            /**
             * As messages are filtered and ordered by timestamp, this function
             * can be implemented by any subclass to perform additional
             * processing.
             */
            virtual void
            parse_individual_message(const nlohmann::json& json_message);

            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            std::map<std::string, nlohmann::json>
            filter(const std::string& messages_filepath) override;

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
            // Indicates whether a message informing about the mission start was
            // received. Messages received before the mission starts will be
            // ignored.
            bool mission_started = false;

            int elapsed_seconds = 0;

            time_t mission_initial_timestamp;

            int mission_trial_number = -1;

            std::string experiment_id;

            std::unordered_map<std::string, std::string> fov_filepaths;
        };

    } // namespace model
} // namespace tomcat
