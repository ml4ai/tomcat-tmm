#include "EvidenceSet.h"

#include <iomanip>

#include <boost/filesystem.hpp>

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
        EvidenceSet::EvidenceSet(bool event_based) : event_based(event_based) {}

        EvidenceSet::EvidenceSet(const string& data_folder_path,
                                 bool event_based)
            : event_based(event_based) {
            this->init_from_folder(data_folder_path);
        }

        EvidenceSet::EvidenceSet(
            const vector<vector<nlohmann::json>>& new_dict_like_data,
            bool event_based)
            : event_based(event_based) {

            this->set_dict_like_data(new_dict_like_data);
        }

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
                    if (filename == METADATA_FILE) {
                        fstream log_file;
                        log_file.open(file.path().string());
                        if (log_file.is_open()) {
                            this->metadata = nlohmann::json::parse(log_file);
                            log_file.close();
                        }
                    }
                    else if (filename == TIME_2_EVENT_MAP_FILE) {
                        this->event_based = true;
                        fstream mapping_file;
                        mapping_file.open(file.path().string());
                        if (mapping_file.is_open()) {
                            nlohmann::json map_per_point =
                                nlohmann::json::parse(mapping_file);

                            for (const auto& mappings : map_per_point) {
                                set<pair<int, int>> time_2_event;
                                for (const auto& mapping : mappings) {
                                    int time_step = mapping["time_step"];
                                    int event_idx = mapping["event"];
                                    time_2_event.insert(
                                        {-time_step, event_idx});
                                }
                                this->time_2_event_per_data_point.push_back(
                                    time_2_event);
                            }
                            mapping_file.close();
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
                    else if (filename == DICT_LIKE_DATA_FILE) {
                        fstream log_file;
                        log_file.open(file.path().string());
                        if (log_file.is_open()) {
                            const auto& json_dict_data =
                                nlohmann::json::parse(log_file);
                            vector<vector<nlohmann::json>> new_dict_like_data;
                            for (const auto& json_series :
                                 json_dict_data["data_points"]) {
                                vector<nlohmann::json> series;
                                for (const auto& dict_like_single_data :
                                     json_series["data_series"]) {
                                    series.push_back(dict_like_single_data);
                                }
                                new_dict_like_data.push_back(series);
                            }
                            this->set_dict_like_data(new_dict_like_data);
                            log_file.close();
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
            if (this->get_time_steps() <= time_step + 1)
                return;

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

            this->save_matrix_data(output_dir);
            this->save_metadata(output_dir);
            this->save_event_mapping(output_dir);
            this->save_dict_like_data(output_dir);
        }

        void
        EvidenceSet::save_matrix_data(const std::string& output_dir) const {
            for (auto& [node_label, data] : this->node_label_to_data) {
                string filename = node_label;
                string filepath = get_filepath(output_dir, filename);
                save_tensor_to_file(filepath, data);
            }
        }

        void EvidenceSet::save_metadata(const std::string& output_dir) const {
            if (!this->metadata.empty()) {
                string metadata_filepath =
                    get_filepath(output_dir, METADATA_FILE);
                ofstream mmetadata_file;
                mmetadata_file.open(metadata_filepath);
                mmetadata_file << setw(4) << this->metadata;
                mmetadata_file.close();
            }
        }

        void
        EvidenceSet::save_event_mapping(const std::string& output_dir) const {
            if (this->event_based) {
                nlohmann::json map_per_point = nlohmann::json::array();
                for (const auto& time_2_events :
                     this->time_2_event_per_data_point) {
                    nlohmann::json mappings = nlohmann::json::array();
                    for (const auto [neg_time_step, event_idx] :
                         time_2_events) {
                        nlohmann::json mapping;
                        mapping["time_step"] = -neg_time_step;
                        mapping["event"] = event_idx;
                        mappings.push_back(mapping);
                    }
                    map_per_point.push_back(mappings);
                }

                string mapping_filepath =
                    get_filepath(output_dir, TIME_2_EVENT_MAP_FILE);
                ofstream mapping_file;
                mapping_file.open(mapping_filepath);
                mapping_file << setw(4) << map_per_point;
                mapping_file.close();
            }
        }

        void
        EvidenceSet::save_dict_like_data(const std::string& output_dir) const {
            // First we embed the vector structure to an array inside a json.
            if (!this->dict_like_data.empty()) {
                nlohmann::json json_dict_data;
                json_dict_data["data_points"] = nlohmann::json::array();
                int i = 0;
                for (const auto& dict_data_per_time : this->dict_like_data) {
                    nlohmann::json json_series;
                    json_series["data_series"] = nlohmann::json::array();

                    for (const auto& single_dict_data : dict_data_per_time) {
                        json_series["data_series"].push_back(single_dict_data);
                    }

                    json_dict_data["data_points"].push_back(json_series);
                }

                string filepath = get_filepath(output_dir, DICT_LIKE_DATA_FILE);
                ofstream file;
                file.open(filepath);
                file << setw(4) << json_dict_data;
                file.close();
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

            // Merge dict-like data
            if (this->dict_like_data.empty()) {
                this->dict_like_data = other.dict_like_data;
                this->time_steps = other.time_steps;
            }
            else {
                this->dict_like_data.insert(this->dict_like_data.end(),
                                            other.get_dict_like_data().begin(),
                                            other.get_dict_like_data().end());
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

            // Merge dict-like data
            if (this->dict_like_data.empty()) {
                this->dict_like_data = other.dict_like_data;
                this->num_data_points = other.num_data_points;
            }
            else {
                for (int i = 0; i < this->dict_like_data.size(); i++) {
                    this->dict_like_data[i].insert(
                        this->dict_like_data[i].end(),
                        other.get_dict_like_data()[i].begin(),
                        other.get_dict_like_data()[i].end());
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

            new_set.event_based = this->event_based;
            if (!this->time_2_event_per_data_point.empty()) {
                new_set.time_2_event_per_data_point =
                    vector<set<pair<int, int>>>(
                        1, this->time_2_event_per_data_point[data_point_idx]);
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

        int EvidenceSet::get_column_index_for(int data_point,
                                              int time_step) const {
            int col_idx = time_step;

            if (this->event_based) {
                auto it =
                    this->time_2_event_per_data_point[data_point].lower_bound(
                        {-time_step, 0});
                if (it == this->time_2_event_per_data_point[data_point].end()) {
                    col_idx = 0;
                }
                else {
                    col_idx = it->second;
                }
            }

            return col_idx;
        }

        int EvidenceSet::get_num_events_for(int data_point) const {
            int num_events = this->get_time_steps();
            if (this->event_based) {
                num_events =
                    this->time_2_event_per_data_point[data_point].size();
            }

            return num_events;
        }

        void EvidenceSet::merge(const EvidenceSet& other_set) {
            this->metadata.insert(this->metadata.end(),
                                  other_set.get_metadata().begin(),
                                  other_set.get_metadata().end());
            this->vstack(other_set);
        }

        void EvidenceSet::set_dict_like_data(
            const vector<vector<nlohmann::json>>& new_dict_like_data) {
            this->dict_like_data = new_dict_like_data;

            this->num_data_points = (int)new_dict_like_data.size();
            this->time_steps = (int)new_dict_like_data.size() > 0
                                   ? (int)new_dict_like_data[0].size()
                                   : 0;
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

        void EvidenceSet::set_time_2_event_per_data_point(
            const vector<vector<pair<int, int>>>& time_2_event_per_data_point) {

            this->time_2_event_per_data_point.resize(
                time_2_event_per_data_point.size());
            for (int d = 0; d < this->time_2_event_per_data_point.size(); d++) {
                for (auto [time_step, event_idx] :
                     time_2_event_per_data_point[d]) {
                    // We insert the negative of the time step so we can use the
                    // lower_bound function of the set data structure to get the
                    // least time step.
                    this->time_2_event_per_data_point[d].insert(
                        {-time_step, event_idx});
                }
            }
        }

        bool EvidenceSet::is_event_based() const { return event_based; }

        const vector<vector<nlohmann::json>>&
        EvidenceSet::get_dict_like_data() const {
            return dict_like_data;
        }

    } // namespace model
} // namespace tomcat
