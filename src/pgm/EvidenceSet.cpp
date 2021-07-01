#include "EvidenceSet.h"

#include <boost/filesystem.hpp>
#include <converter/MessageConverter.h>

#include "utils/EigenExtensions.h"
#include "utils/FileHandler.h"
#include "utils/Tensor3.h"

namespace tomcat {
    namespace model {

        using namespace std;
        namespace fs = boost::filesystem;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        EvidenceSet::EvidenceSet() {}

        EvidenceSet::EvidenceSet(const string& data_folder_path) {
            this->init_from_folder(data_folder_path);
        }

        EvidenceSet::~EvidenceSet() {}

        //----------------------------------------------------------------------
        // Operator overload
        //----------------------------------------------------------------------
        ostream& operator<<(ostream& os, const EvidenceSet& set) {
            for (const auto& [label, data] : set.node_label_to_data) {
                os << label << "\n";
                os << data;
            }

            return os;
        }

        const Tensor3& EvidenceSet::operator[](const string& node_label) const {
            return this->node_label_to_data.at(node_label);
        }

        const Tensor3& EvidenceSet::operator[](string&& node_label) const {
            return this->node_label_to_data.at(node_label);
        }

        //----------------------------------------------------------------------
        // Static functions
        //----------------------------------------------------------------------
        Eigen::MatrixXd EvidenceSet::get_observations_in_window(
            const Tensor3& data,
            const Eigen::VectorXd& assignment,
            int window) {

            Eigen::MatrixXd logical_data = (data == assignment);
            int first_time_step = get_first_time_with_observation(data);
            int num_rows = logical_data.rows();
            int num_cols = logical_data.cols();
            // Replace the first columns with no observable data with NO_OBS
            // value so it can be preserved in the operations below.
            logical_data.block(0, 0, num_rows, first_time_step) =
                Eigen::MatrixXd::Constant(num_rows, first_time_step, NO_OBS);

            Eigen::MatrixXd logical_data_in_window(num_rows, num_cols - window);
            if (window == 0) {
                logical_data_in_window = logical_data;
            }
            else {
                Eigen::MatrixXd cum_sum_over_time = cum_sum(logical_data, 1);

                Eigen::MatrixXd num_obs_in_window(num_rows, num_cols - window);
                int j = 0;
                for (int w = window; w < cum_sum_over_time.cols(); w++) {
                    num_obs_in_window.col(j) =
                        cum_sum_over_time.col(w).array() -
                        cum_sum_over_time.col(j).array();
                    j++;
                }

                Eigen::MatrixXd ones = Eigen::MatrixXd::Ones(
                    num_obs_in_window.rows(), num_obs_in_window.cols());
                Eigen::MatrixXd zeros = Eigen::MatrixXd::Zero(
                    num_obs_in_window.rows(), num_obs_in_window.cols());
                Eigen::MatrixXd no_obs = Eigen::MatrixXd::Constant(
                    num_obs_in_window.rows(), num_obs_in_window.cols(), NO_OBS);
                no_obs = (num_obs_in_window.array() < 0).select(no_obs, zeros);
                logical_data_in_window =
                    (num_obs_in_window.array() > 0).select(ones, no_obs);
            }

            return logical_data_in_window;
        }

        int EvidenceSet::get_first_time_with_observation(const Tensor3& data) {

            int time_step = 0;
            auto [d1, d2, d3] = data.get_shape();
            for (int k = 0; k < d3; k++) {
                bool obs_data = false;
                for (int j = 0; j < d2; j++) {
                    // If every data in depth is non_observable, the given time
                    // step k for the data point in row j is defined as non
                    // observable.
                    for (int i = 0; i < d1; i++) {
                        if (data.at(i, j, k) != NO_OBS) {
                            obs_data = true;
                            break;
                        }
                    }

                    // Also, if at least one data point is non-observable at
                    // given time step, no other data point should be.
                    if (!obs_data) {
                        break;
                    }
                }
                if (obs_data) {
                    break;
                }
                else {
                    time_step++;
                }
            }

            return time_step;
        }

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        void EvidenceSet::init_from_folder(const string& data_folder_path) {
            for (const auto& file : fs::directory_iterator(data_folder_path)) {
                string filename = file.path().filename().string();
                if (fs::is_regular_file(file)) {
                    if (filename == MessageConverter::LOG_FILE) {
                        fstream log_file;
                        log_file.open(file.path().string());
                        if (log_file.is_open()) {
                            this->metadata = nlohmann::json::parse(
                                log_file)["files_converted"];
                        }
                    }
                    else if (file.path().extension() == "") {
                        string node_label = remove_extension(filename);
                        Tensor3 data =
                            read_tensor_from_file(file.path().string());
                        if (!data.is_empty()) {
                            this->add_data(node_label, data);
                        }
                    }
                }
            }
        }

        vector<string> EvidenceSet::get_node_labels() const {
            vector<string> node_labels;
            node_labels.reserve(this->node_label_to_data.size());

            for (auto& [label, data] : this->node_label_to_data) {
                node_labels.push_back(label);
            }

            return node_labels;
        }

        void EvidenceSet::add_data(const string& node_label,
                                   const Tensor3& data,
                                   bool check_dimensions) {
            if (this->num_data_points == 0 && this->time_steps == 0) {
                this->num_data_points = data.get_shape()[1];
                this->time_steps = data.get_shape()[2];
            }
            else {
                if (check_dimensions) {
                    if (data.get_shape()[1] != this->num_data_points) {
                        throw invalid_argument(
                            "The number of data points must be the same for "
                            "all the observable nodes in the folder.");
                    }

                    if (data.get_shape()[2] != this->time_steps) {
                        throw invalid_argument(
                            "The number of time steps must be the same for "
                            "all the observable nodes in the folder.");
                    }
                }
            }

            this->node_label_to_data[node_label] = data;
        }

        bool EvidenceSet::has_data_for(const string& node_label) const {
            return EXISTS(node_label, this->node_label_to_data);
        }

        void EvidenceSet::set_data_for(const string& node_label,
                                       const Tensor3& data) {
            if (!EXISTS(node_label, this->node_label_to_data)) {
                throw TomcatModelException("The node " + node_label +
                                           "does not belong to the DBN Data.");
            }

            this->node_label_to_data[node_label] = data;
        }

        Eigen::MatrixXd EvidenceSet::get_observations_in_window_for(
            const string& node_label,
            const Eigen::VectorXd& assignment,
            int window) const {

            return EvidenceSet::get_observations_in_window(
                this->node_label_to_data.at(node_label), assignment, window);
        }

        void EvidenceSet::keep_first(int num_samples) {
            for (auto& [node_label, data] : this->node_label_to_data) {
                data = data.slice(0, num_samples, 1);
            }
            this->num_data_points = num_samples;
        }

        void EvidenceSet::keep_only(int data_idx) {
            for (auto& [node_label, data] : this->node_label_to_data) {
                data = data.slice(data_idx, data_idx + 1, 1);
            }
            this->num_data_points = 1;
        }

        void EvidenceSet::shrink_up_to(int time_step) {
            for (auto& [node_label, data] : this->node_label_to_data) {
                data = data.slice(0, time_step + 1, 2);
            }
            this->time_steps = time_step + 1;
        }

        bool EvidenceSet::empty() const { return this->num_data_points == 0; }

        void EvidenceSet::remove(const string& node_label) {
            this->node_label_to_data.erase(node_label);
        }

        void EvidenceSet::save(const string& output_dir) const {
            fs::create_directories(output_dir);
            for (auto& [node_label, data] : this->node_label_to_data) {
                string filename = node_label;
                string filepath = get_filepath(output_dir, filename);
                save_tensor_to_file(filepath, data);
            }
        }

        EvidenceSet EvidenceSet::at(int row, int col) const {
            EvidenceSet small_set;

            for (const auto [label, data] : node_label_to_data) {
                small_set.add_data(label, Tensor3(data.at(row, col)));
            }

            return small_set;
        }

        void EvidenceSet::vstack(const EvidenceSet& other) {
            for (const auto& label : other.get_node_labels()) {
                if (this->has_data_for(label)) {
                    this->node_label_to_data.at(label).vstack(
                        other.node_label_to_data.at(label));
                }
                else {
                    this->node_label_to_data[label] =
                        other.node_label_to_data.at(label);
                    this->time_steps = other.time_steps;
                }
            }
            this->num_data_points += other.num_data_points;
        }

        void EvidenceSet::hstack(const EvidenceSet& other) {
            for (const auto& label : other.get_node_labels()) {
                if (this->has_data_for(label)) {
                    this->node_label_to_data.at(label).hstack(
                        other.node_label_to_data.at(label));
                }
                else {
                    this->node_label_to_data[label] =
                        other.node_label_to_data.at(label);
                    this->num_data_points = other.num_data_points;
                }
            }
            this->time_steps += other.get_time_steps();
        }

        EvidenceSet
        EvidenceSet::get_single_point_data(int data_point_idx) const {
            EvidenceSet new_set;
            for (const auto& [node_label, data] : this->node_label_to_data) {
                new_set.add_data(node_label, data.row(data_point_idx));
            }

            return new_set;
        }

        EvidenceSet EvidenceSet::get_single_time_data(int time_step) const {
            EvidenceSet new_set;
            for (const auto& [node_label, data] : this->node_label_to_data) {
                new_set.add_data(node_label, data.col(time_step));
            }

            return new_set;
        }

        //----------------------------------------------------------------------
        // Getters & Setters
        //----------------------------------------------------------------------
        int EvidenceSet::get_num_data_points() const { return num_data_points; }

        int EvidenceSet::get_time_steps() const { return time_steps; }

        void EvidenceSet::set_metadata(const nlohmann::json& metadata) {
            this->metadata = metadata;
        }

        const nlohmann::json& EvidenceSet::get_metadata() const {
            return metadata;
        }

    } // namespace model
} // namespace tomcat
