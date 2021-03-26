#include "DataSplitter.h"

#include <algorithm>
#include <iomanip>

#include <fmt/format.h>
#include <nlohmann/json.hpp>
#include <boost/filesystem.hpp>

#include <gsl/gsl_randist.h>

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        DataSplitter::DataSplitter() {}

        DataSplitter::DataSplitter(
            const EvidenceSet& data,
            int num_folds,
            const shared_ptr<gsl_rng>& random_generator) {

            this->define_indices(
                random_generator, data.get_num_data_points(), num_folds);
            this->split(data);
        }

        DataSplitter::DataSplitter(
            const EvidenceSet& data,
            double test_prop,
            const shared_ptr<gsl_rng>& random_generator) {

            this->define_indices(
                random_generator, data.get_num_data_points(), test_prop);
            this->split(data);
        }

        DataSplitter::DataSplitter(const EvidenceSet& training_data,
                                   const EvidenceSet& test_data) {

            this->splits.push_back(make_pair(training_data, test_data));
        }

        DataSplitter::DataSplitter(const EvidenceSet& data,
                                   const std::string& indices_dir) {
            this->load_indices(indices_dir);
            this->split(data);
        }

        DataSplitter::~DataSplitter() {}

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        void DataSplitter::define_indices(
            const shared_ptr<gsl_rng>& random_generator,
            int data_size,
            int num_folds) {

            if (num_folds > data_size) {
                throw invalid_argument(
                    "The number of folds must be at most the number of "
                    "data points in the evidence set.");
            }

            vector<int> fold_sizes = this->get_fold_sizes(data_size, num_folds);
            vector<int> shuffled_indices =
                this->get_shuffled_indices(data_size, random_generator);

            int start_idx = 0;
            for (int i = 0; i < num_folds; i++) {
                int end_idx = start_idx + fold_sizes[i] - 1;

                vector<int> training_indices;
                if (start_idx > 0) {
                    training_indices.insert(training_indices.end(),
                                            shuffled_indices.begin(),
                                            shuffled_indices.begin() +
                                                start_idx);
                }
                if (end_idx < shuffled_indices.size() - 1) {
                    training_indices.insert(training_indices.end(),
                                            shuffled_indices.begin() + end_idx +
                                                1,
                                            shuffled_indices.end());
                }

                vector<int> test_indices(shuffled_indices.begin() + start_idx,
                                         shuffled_indices.begin() + end_idx +
                                             1);

                // We save these to use in the get_info method.
                this->training_indices_per_fold.push_back(training_indices);
                this->test_indices_per_fold.push_back(test_indices);
                start_idx = end_idx + 1;
            }
        }

        void DataSplitter::define_indices(
            const shared_ptr<gsl_rng>& random_generator,
            int data_size,
            double test_proportion) {

            if (test_proportion < 0 || test_proportion > 1) {
                throw invalid_argument(
                    "The proportion of samples in the test set must be "
                    "between 0 and 1.");
            }

            vector<int> shuffled_indices =
                this->get_shuffled_indices(data_size, random_generator);
            int test_size = data_size * test_proportion;

            this->training_indices_per_fold.resize(0);
            this->test_indices_per_fold.resize(0);

            auto& training_indices = this->training_indices_per_fold.at(0);
            auto& test_indices = this->test_indices_per_fold.at(0);
            test_indices.insert(test_indices.begin(),
                                shuffled_indices.begin(),
                                shuffled_indices.begin() + test_size);
            training_indices.insert(training_indices.begin(),
                                    shuffled_indices.begin() + test_size,
                                    shuffled_indices.end());
        }

        void DataSplitter::load_indices(const std::string& indices_dir) {
            string filepath = fmt::format("{}/indices.json", indices_dir);
            fstream file;
            file.open(filepath);
            if (file.is_open()) {
                nlohmann::json json_indices = nlohmann::json::parse(file);

                int num_folds = json_indices["num_folds"];

                this->training_indices_per_fold.resize(num_folds);
                this->test_indices_per_fold.resize(num_folds);

                int k = 0;
                for (const auto& indices :
                     json_indices["shuffled_training_indices"]) {
                    for (int index : indices) {
                        this->training_indices_per_fold.at(k).push_back(index);
                    }
                    k++;
                }

                k = 0;
                for (const auto& indices :
                     json_indices["shuffled_test_indices"]) {
                    for (int index : indices) {
                        this->test_indices_per_fold.at(k).push_back(index);
                    }
                    k++;
                }
            }
            else {
                stringstream ss;
                ss << "The file " << filepath << " does not exist.";
                throw TomcatModelException(ss.str());
            }
        }

        void DataSplitter::split(const EvidenceSet& data) {
            int num_folds = this->training_indices_per_fold.size();
            this->splits.reserve(num_folds);

            for (int k = 0; k < num_folds; k++) {

                const auto& training_indices =
                    this->training_indices_per_fold.at(k);
                const auto& test_indices = this->test_indices_per_fold.at(k);

                EvidenceSet training;
                EvidenceSet test;
                for (auto& node_label : data.get_node_labels()) {
                    Tensor3 node_data = data[node_label];
                    Tensor3 training_data =
                        node_data.slice(training_indices, 1);
                    Tensor3 test_data = node_data.slice(test_indices, 1);

                    training.add_data(node_label, training_data);
                    test.add_data(node_label, test_data);
                }

                if (num_folds == 1) {
                    // In this case the single fold must be used as training
                    // data instead of test data.
                    this->splits.push_back(make_pair(test, training));
                }
                else {
                    this->splits.push_back(make_pair(training, test));
                }
            }
        }

        vector<int> DataSplitter::get_shuffled_indices(
            int num_data_points,
            const shared_ptr<gsl_rng>& random_generator) const {
            int* indices = new int[num_data_points];
            for (int i = 0; i < num_data_points; i++) {
                indices[i] = i;
            }

            gsl_ran_shuffle(
                random_generator.get(), indices, num_data_points, sizeof(int));

            return vector<int>(indices, indices + num_data_points);
        }

        vector<int> DataSplitter::get_fold_sizes(int num_data_points,
                                                 int num_folds) const {
            int fold_size = floor(num_data_points / num_folds);
            int excess = num_data_points % num_folds;
            vector<int> fold_sizes(num_folds);

            for (int i = 0; i < num_folds; i++) {
                fold_sizes[i] = fold_size;
                if (excess > 0) {
                    fold_sizes[i]++;
                    excess--;
                }
            }

            return fold_sizes;
        }

        void DataSplitter::get_info(nlohmann::json& json) const {
            json["num_folds"] = this->splits.size();

            if (!this->training_indices_per_fold.empty()) {
                json["shuffled_training_indices"] = nlohmann::json::array();
                for (const auto& training_indices :
                     this->training_indices_per_fold) {
                    nlohmann::json json_indices = nlohmann::json::array();
                    for (int idx : training_indices) {
                        json_indices.push_back(idx);
                    }
                    json["shuffled_training_indices"].push_back(json_indices);
                }
            }

            if (!this->test_indices_per_fold.empty()) {
                json["shuffled_test_indices"] = nlohmann::json::array();
                for (const auto& test_indices : this->test_indices_per_fold) {
                    nlohmann::json json_indices = nlohmann::json::array();
                    for (int idx : test_indices) {
                        json_indices.push_back(idx);
                    }
                    json["shuffled_test_indices"].push_back(json_indices);
                }
            }
        }

        void DataSplitter::save_indices(const std::string& indices_dir) const {
            boost::filesystem::create_directories(indices_dir);

            string filepath = fmt::format("{}/indices.json", indices_dir);
            ofstream indices_file;
            indices_file.open(filepath);

            nlohmann::json indices_json;
            this->get_info(indices_json);

            indices_file << setw(4) << indices_json;
            indices_file.close();
        }

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        const vector<DataSplitter::Split>& DataSplitter::get_splits() const {
            return splits;
        }

    } // namespace model
} // namespace tomcat
