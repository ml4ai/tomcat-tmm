#pragma once

#include <memory>
#include <utility>
#include <vector>

#include <gsl/gsl_rng.h>
#include <nlohmann/json.hpp>

#include "pgm/EvidenceSet.h"
#include "utils/Definitions.h"

namespace tomcat {
    namespace model {

        /**
         * This class is responsible for splitting data into disjoint folds
         * where k-1 of them is used for training data and the remaining one for
         * test data creating a list of this pair of data sets by repeating this
         * logic for all the k folds.
         */
        class DataSplitter {
          public:
            //------------------------------------------------------------------
            // Types, Enums & Constants
            //------------------------------------------------------------------
            typedef std::pair<EvidenceSet, EvidenceSet> Split;

            //------------------------------------------------------------------
            // Constructors & Destructor
            //------------------------------------------------------------------

            /**
             * Creates an instance of a KFold data splitter.
             */
            DataSplitter();

            /**
             * Creates an instance of a KFold data splitter.
             *
             * @param data: data to be split
             * @param num_folds: number of splits
             * @param random_generator: random number generator
             */
            DataSplitter(const EvidenceSet& data,
                         int num_folds,
                         const std::shared_ptr<gsl_rng>& random_generator);

            /**
             * Creates an instance of a KFold data splitter.
             *
             * @param data: data to be split
             * @param test_prop: proportion of samples in the test set
             * @param random_generator: random number generator
             */
            DataSplitter(const EvidenceSet& data,
                         double test_prop,
                         const std::shared_ptr<gsl_rng>& random_generator);

            /**
             * Creates an instance of a KFold data splitter with a
             * pre-defined training and test data.
             *
             * @param training_data: training data
             * @param test_data: test data
             */
            DataSplitter(const EvidenceSet& training_data,
                         const EvidenceSet& test_data);

            /**
             * Creates an instance of a KFold data splitter with split
             * determined by an indices file.
             *
             * @param data: data to be split
             * @param indices_dir: directory where the json file containing
             * information about how the data must be split is located. The
             * file must have the same format as the one generated by the
             * function save_indices. This constructor can be used to evaluate a
             * pre-trained model for which training was performed in multiple
             * folds.
             */
            DataSplitter(const EvidenceSet& data,
                         const std::string& indices_dir);

            ~DataSplitter();

            //------------------------------------------------------------------
            // Copy & Move constructors/assignments
            //------------------------------------------------------------------
            DataSplitter(const DataSplitter&) = default;

            DataSplitter& operator=(const DataSplitter&) = default;

            DataSplitter(DataSplitter&&) = default;

            DataSplitter& operator=(DataSplitter&&) = default;

            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            /**
             * Writes information about the splitter in a json object.
             *
             * @param json: json object
             */
            void get_info(nlohmann::json& json) const;

            /**
             * Save training and test data point indices to a json file in a
             * given folder.
             */
            void save_indices(const std::string& indices_dir) const;

            //------------------------------------------------------------------
            // Getters & Setters
            //------------------------------------------------------------------
            const std::vector<Split>& get_splits() const;

          private:
            //------------------------------------------------------------------
            // Member functions
            //------------------------------------------------------------------

            /**
             * Shuffles and split training and test indices based on a split
             * per fixed number of folds.
             *
             * @param random_generator: random number generator
             * @param data_size: number of data points in the data set
             * @param num_folds: number of folds to split the data into
             */
            void
            define_indices(const std::shared_ptr<gsl_rng>& random_generator,
                           int data_size,
                           int num_folds);

            /**
             * Shuffles and split training and test indices using the
             * test_proportion attribute to determine the size of the test set.
             *
             * @param random_generator: random number generator
             * @param data_size: number of data points in the data set
             * @param test_proportion: proportion of the data that must
             * comprise the test set
             */
            void
            define_indices(const std::shared_ptr<gsl_rng>& random_generator,
                           int data_size,
                           double test_proportion);

            /**
             * Loads training and test indices from a json-formatted indices
             * file.
             *
             * @param indices_dir: directory where the json file containing
             * pre-defined indices for the training and test data folds is
             * located.
             */
            void load_indices(const std::string& indices_dir);

            /**
             * Creates K data splits comprised of disjoint training and test
             * set. This method assumes the indices were previously defined,
             * either by calling the function define_indices or by leading
             * the indices from an indices file.
             *
             * @param data: data to be split
             */
            void split(const EvidenceSet& data);

            /**
             * Returns a list of shuffled indices of the data points.
             *
             * @param num_data_points: number of data points that will be
             * split into folds
             * @param random_generator: random number generator
             *
             * @return Shuffled indices of the data points.
             */
            std::vector<int> get_shuffled_indices(
                int num_data_points,
                const std::shared_ptr<gsl_rng>& random_generator) const;

            /**
             * Returns the number of data points in each fold.
             *
             * @param num_data_points: number of data points that will be
             * split into folds
             * @param num_folds: number of splits
             *
             * @return Number of data points in each one of the folds.
             */
            std::vector<int> get_fold_sizes(int num_data_points,
                                            int num_folds) const;

            //------------------------------------------------------------------
            // Data members
            //------------------------------------------------------------------

            // Data split in training and test sets. Populated in the creation
            // of the class object.
            std::vector<Split> splits;

            std::vector<std::vector<int>> training_indices_per_fold;

            std::vector<std::vector<int>> test_indices_per_fold;
        };

    } // namespace model
} // namespace tomcat
