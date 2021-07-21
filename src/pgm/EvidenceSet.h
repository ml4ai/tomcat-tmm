#pragma once

#include <set>
#include <string>
#include <unordered_map>
#include <utility>

#include <nlohmann/json.hpp>

#include "utils/Definitions.h"
#include "utils/FileHandler.h"

namespace tomcat {
    namespace model {

        /**
         * This class contains a map between node labels in a DBN and values
         * stored into 3-dimensional tensors (value dimensionality, number data
         * points, time steps).
         */
        class EvidenceSet {
          public:
            //------------------------------------------------------------------
            // Constructors & Destructor
            //------------------------------------------------------------------

            /**
             * Creates a blank DBNData object.
             *
             * @param event_based: whether the data is based on events or time
             * steps.
             */
            EvidenceSet(bool event_based = false);

            /**
             * Creates an DBNData object with data from files in a given folder.
             *
             * @param data_folder_path: folder where files with nodes'
             * values are stored.
             * @param event_based: whether the data is based on events or time
             * steps.
             */
            EvidenceSet(const std::string& data_folder_path,
                        bool event_based = false);

            ~EvidenceSet();

            //------------------------------------------------------------------
            // Copy & Move constructors/assignments
            //------------------------------------------------------------------
            EvidenceSet(const EvidenceSet&) = default;

            EvidenceSet& operator=(const EvidenceSet&) = default;

            EvidenceSet(EvidenceSet&&) = default;

            EvidenceSet& operator=(EvidenceSet&&) = default;

            //------------------------------------------------------------------
            // Operator overload
            //------------------------------------------------------------------
            friend std::ostream& operator<<(std::ostream& os,
                                            const EvidenceSet& set);

            /**
             * Returns data for a given node.
             *
             * @param node_label: node's label
             *
             * @return Values for the informed node.
             */
            const Tensor3& operator[](const std::string& node_label) const;

            /**
             * Returns data for a given node.
             *
             * @param node_label: node's label
             *
             * @return Values for the informed node.
             */
            const Tensor3& operator[](std::string&& node_label) const;

            //------------------------------------------------------------------
            // Static functions
            //------------------------------------------------------------------

            /**
             * Returns a matrix such that, for each coefficient in the original
             * matrix up to the original number of columns - window, 1 will be
             * assigned to the coefficient if a given assignment was observed
             * within a given window (if the assignment shows up in at least
             * one of the subsequent columns of the coefficient up to window
             * size), 0 otherwise. The assignment is compared with the elements
             * in the first dimension of the tensor. Therefore the result will
             * be a matrix.
             *
             * @param data: data
             * @param assignment: assignment to compare against
             * @param window: determines the number of columns to look ahead (it
             * only looks ahead for windows of size > 1)
             *
             * @return Logical matrix with observations within a window.
             */
            static Eigen::MatrixXd
            get_observations_in_window(const Tensor3& data,
                                       const Eigen::VectorXd& assignment,
                                       int window);
            /**
             * Returns the first time step with values different than NO_OBS for
             * some node's data.
             *
             * @param data: data
             *
             * @return First time step with actual data.
             */
            static int get_first_time_with_observation(const Tensor3& data);

            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            /**
             * Returns the labels of the nodes which the DBNData has values
             * for.
             *
             * @return Nodes' labels.
             */
            std::vector<std::string> get_node_labels() const;

            /**
             * Adds data for a specific node.
             *
             * @param node_label: node's label
             * @param data: values for the node
             * @param check_dimensions: whether the number of rows and columns
             * have to be the same for all nodes in the evidence set
             */
            void add_data(const std::string& node_label,
                          const Tensor3& data,
                          bool check_dimensions = true);

            /**
             * Checks whether this object contains data for a given node.
             *
             * @param node_label: node's label
             *
             * @return: Whether this object contains data for a given node.
             */
            bool has_data_for(const std::string& node_label) const;

            /**
             * Assigns a new tensor to a node;
             *
             * @param node_label: node's label
             *
             */
            void set_data_for(const std::string& node_label,
                              const Tensor3& data);

            /**
             * For a given node's data, returns a logical matrix flagging the
             * time steps where a given assignment was observed within a given
             * window.
             *
             * @param node_label: node's label
             * @param assignment: assignment to compare against
             * @param window: determines the number of columns to look ahead (it
             * only looks ahead for windows of size > 1)
             *
             * @return Logical matrix with observations within a window.
             */
            Eigen::MatrixXd
            get_observations_in_window_for(const std::string& node_label,
                                           const Eigen::VectorXd& assignment,
                                           int window) const;

            /**
             * Keep the first samples and remove the rest.
             *
             * @param num_samples: Number of the first samples to preserve.
             */
            void keep_first(int num_samples);

            /**
             * Keep the row at the informed index and remove all others.
             *
             * @param data_idx: data index
             */
            void keep_only(int data_idx);

            /**
             * Shrinks the data up to a time slice (inclusive).
             *
             * @param time_slice: max time step included in the data.
             */
            void shrink_up_to(int time_step);

            /**
             * Indicates whether the set has data or not.
             *
             * @return True if the set has any data point.
             */
            bool empty() const;

            /**
             * Removes data for a given node.
             *
             * @param node_label: nodes' label
             */
            void remove(const std::string& node_label);

            /**
             * Saves content of the set to a folder.
             *
             * @param output_dir: directory where the data must be saved.
             */
            void save(const std::string& output_dir) const;

            /**
             * Retrieves an evidence set comprised by tensors with values
             * from the row and column informed preserved (axis 1 and 2 of
             * the tensor)
             *
             * @param row: matrix row to select
             * @param col: matrix column to select
             *
             * @return Single point and time evidence set
             */
            EvidenceSet at(int row, int col) const;

            /**
             * Appends the content of another set into this set along the
             * second dimension of the tensors.
             *
             * @param other: data to append
             */
            void vstack(const EvidenceSet& other);

            /**
             * Appends the content of another set into this set along the
             * third dimension of the tensors.
             *
             * @param other: data to append
             */
            void hstack(const EvidenceSet& other);

            /**
             * Gets only data for a single data point
             *
             * @param data_point_idx: index of the data point
             *
             * @return Data
             */
            EvidenceSet get_single_point_data(int data_point_idx) const;

            /**
             * Gets only data for a single time step
             *
             * @param time_step: time step to select data from
             *
             * @return Data
             */
            EvidenceSet get_single_time_data(int time_step) const;

            /**
             * Returns the index of the column related to a specific time step.
             * If the evidence set contains time-based data, the column index is
             * the time step itself, otherwise, the column index is the most
             * recent event before or at the same time step informed
             *
             * @param data_point: index of the data point
             * @param time_step: time step
             * @return
             */
            int get_column_index_for(int data_point, int time_step) const;

            /**
             * For a given data point, returns the number of valid events. If
             * the set is not event based, the number corresponds to the number
             * of time steps in the whole set.
             *
             * @param data_point: data point index
             * @return
             */
            int get_num_events_for(int data_point) const;

            //------------------------------------------------------------------
            // Getters & Setters
            //------------------------------------------------------------------
            int get_num_data_points() const;

            int get_time_steps() const;

            void set_metadata(const nlohmann::json& metadata);

            const nlohmann::json& get_metadata() const;

            void set_time_2_event_per_data_point(
                const std::vector<std::vector<std::pair<int, int>>>&
                    time_2_event_per_data_point);

            bool is_event_based() const;

          private:
            inline static std::string TIME_2_EVENT_MAP_FILE =
                "time_2_event_map.json";

            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            /**
             * Reads data from files in a folder and store in this object.
             *
             * @param data_folder_path: folder where the data files are stored
             */
            void init_from_folder(const std::string& data_folder_path);

            //------------------------------------------------------------------
            // Data members
            //------------------------------------------------------------------
            int num_data_points = 0;

            int time_steps = 0;

            std::unordered_map<std::string, Tensor3> node_label_to_data;

            // Any relevant information regarding the evidence can be added
            // here
            nlohmann::json metadata;

            bool event_based;

            std::vector<std::set<std::pair<int, int>>>
                time_2_event_per_data_point;
        };

    } // namespace model
} // namespace tomcat
