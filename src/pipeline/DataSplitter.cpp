#include "DataSplitter.h"

#include <algorithm>

#include <gsl/gsl_randist.h>

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        DataSplitter::DataSplitter(const EvidenceSet& data,
                                   int num_folds,
                                   shared_ptr<gsl_rng> random_generator) {

            this->split(data, num_folds, random_generator);
        }

        DataSplitter::DataSplitter(const EvidenceSet& data,
                                   float test_prop,
                                   shared_ptr<gsl_rng> random_generator) {

            this->split(data, test_prop, random_generator);
        }

        DataSplitter::DataSplitter(const EvidenceSet& training_data,
                                   const EvidenceSet& test_data) {

            this->splits.push_back(make_pair(training_data, test_data));
        }

        DataSplitter::~DataSplitter() {}

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        void DataSplitter::split(const EvidenceSet& data,
                                 int num_folds,
                                 shared_ptr<gsl_rng> random_generator) {

            if (num_folds > data.get_num_data_points()) {
                throw invalid_argument(
                    "The number of folds is bigger than the number of data "
                    "points in the evidence set.");
            }

            vector<int> fold_sizes =
                this->get_fold_sizes(data.get_num_data_points(), num_folds);
            vector<int> shuffled_indices = this->get_shuffled_indices(
                data.get_num_data_points(), random_generator);

            this->splits.reserve(num_folds);
            int start_idx = 0;

            for (int i = 0; i < num_folds; i++) {
                int end_idx = start_idx + fold_sizes[i] - 1;

                EvidenceSet training;
                EvidenceSet test;
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

                // We save this to use in the get_info method.
                this->test_indices_per_fold.push_back(test_indices);

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

                start_idx = end_idx + 1;
            }
        }

        vector<int> DataSplitter::get_shuffled_indices(
            int num_data_points, shared_ptr<gsl_rng> random_generator) const {
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

        void DataSplitter::split(const EvidenceSet& data,
                                 float test_prop,
                                 shared_ptr<gsl_rng> random_generator) {

            if (test_prop < 0 || test_prop > 1) {
                throw invalid_argument(
                    "The proportion of samples in the test set must be "
                    "between 0 and 1.");
            }

            vector<int> shuffled_indices = this->get_shuffled_indices(
                data.get_num_data_points(), random_generator);
            int test_size = data.get_num_data_points() * test_prop;

            EvidenceSet training;
            EvidenceSet test;
            vector<int> training_indices;
            vector<int> test_indices;
            test_indices.insert(test_indices.begin(),
                shuffled_indices.begin(),
                shuffled_indices.begin() + test_size);
            training_indices.insert(training_indices.begin(),
                                    shuffled_indices.begin() + test_size,
                                    shuffled_indices.end());

            for (auto& node_label : data.get_node_labels()) {
                Tensor3 node_data = data[node_label];
                Tensor3 training_data =
                    node_data.slice(training_indices, 1);
                Tensor3 test_data = node_data.slice(test_indices, 1);

                training.add_data(node_label, training_data);
                test.add_data(node_label, test_data);
            }

            this->splits.push_back(make_pair(training, test));
        }

        void DataSplitter::get_info(nlohmann::json& json) const {
            json["num_folds"] = this->splits.size();

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

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        const vector<DataSplitter::Split>& DataSplitter::get_splits() const {
            return splits;
        }

    } // namespace model
} // namespace tomcat
