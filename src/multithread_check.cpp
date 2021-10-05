#include <cstdlib>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include <eigen3/Eigen/Dense>
#include <gsl/gsl_randist.h>

#include "utils/Definitions.h"
#include "utils/Multithreading.h"
#include "multithread/ThreadPool.h"

using namespace std;
using namespace tomcat::model;
using namespace multithread;

vector<double> pdfs = {0.2, 0.1, 0.3, 0.15, 0.25};

Eigen::MatrixXd big_matrix;
Eigen::MatrixXd weights;

// Compute particle weights
void weigh(const pair<int, int>& block) {
    int initial_row = block.first;
    int num_rows = block.second;
    auto weight_patch = weights.block(initial_row, 0, num_rows, 1);
    auto matrix_patch = big_matrix.block(initial_row, 0, num_rows, 1);

    for (int i = 0; i < num_rows; i++) {
        weight_patch(i, 0) = pdfs[matrix_patch(i, 0)];
    }
}

// Replace values in the matrix by random numbers
void elapse(const pair<int, int>& block) {
    int initial_row = block.first;
    int num_rows = block.second;
    auto patch = big_matrix.block(initial_row, 0, num_rows, 1);

    for (int i = 0; i < num_rows; i++) {
        patch(i, 0) = i % 5;
    }

    weigh(block);
}

// Resample particles
void resample(const pair<int, int>& block,
              shared_ptr<gsl_rng>& gen,
              shared_ptr<gsl_ran_discrete_t> ptable) {
    int initial_row = block.first;
    int num_rows = block.second;

    auto patch = big_matrix.block(initial_row, 0, num_rows, 1);

    //    int k = weights.size();
    //    const double* parameters_ptr = weights.col(0).data();
    //    unsigned int* sample_ptr = new unsigned int[k];
    //    gsl_ran_discrete_t* ptable2 =
    //        gsl_ran_discrete_preproc(weights.rows(), weights.col(0).data());

    for (int i = 0; i < num_rows; i++) {
        //        gsl_ran_multinomial(gen.get(), k, 1, parameters_ptr,
        //        sample_ptr); int p = distance(sample_ptr, find(sample_ptr,
        //        sample_ptr + k, 1));
        int p = gsl_ran_discrete(gen.get(), ptable.get());
        //        patch(i, 0) = big_matrix(p, 0);
    }

    //    delete[] sample_ptr;
}

void eval_full_filter(bool col_major_storage, int n) {
    int jobs = 10;
    int T = 1000;

    if (col_major_storage) {
        big_matrix = Eigen::MatrixXd::Random(n, 1);
        weights = Eigen::MatrixXd ::Zero(n, 1);
    }
    else {
        big_matrix = Eigen::Matrix<double,
                                   Eigen::Dynamic,
                                   Eigen::Dynamic,
                                   Eigen::RowMajor>::Random(n, 1);
        weights = Eigen::Matrix<double,
                                Eigen::Dynamic,
                                Eigen::Dynamic,
                                Eigen::RowMajor>::Zero(n, 1);
    }
    shared_ptr<gsl_rng> gen(gsl_rng_alloc(gsl_rng_mt19937));

    for (int j = 1; j <= 10; j++) {
        cout << j << " thread(s): ";
        auto blocks = get_parallel_processing_blocks(j, n);
        Timer t;

        if (j == 1) {
            for (int t = 0; t < T; t++) {
                elapse(blocks[0]);
                shared_ptr<gsl_ran_discrete_t> ptable(gsl_ran_discrete_preproc(
                    weights.rows(), weights.col(0).data()));
                resample(blocks[0], gen, ptable);
            }
        }
        else {
            ThreadPool pool(j);
            vector<future<void>> futures(j);
            auto gens = split_random_generator(gen, j);

            for (int t = 0; t < T; t++) {
                for (int i = 0; i < blocks.size(); i++) {
                    futures[i] = pool.submit(bind(elapse, blocks[i]));
                }

                for (auto& future : futures) {
                    future.get();
                }

                // Prepare the discrete distribution to be sampled.
                shared_ptr<gsl_ran_discrete_t> ptable(gsl_ran_discrete_preproc(
                    weights.rows(), weights.col(0).data()));

                for (int i = 0; i < blocks.size(); i++) {
                    futures[i] =
                        pool.submit(bind(resample, blocks[i], gens[i], ptable));
                }
            }

            for (auto& future : futures) {
                future.get();
            }
        }
    }
}

void complete_elapse(const pair<int, int>& block, int T) {
    for (int t = 0; t < T; t++) {
        elapse(block);
    }
}

void eval_without_pool(bool col_major_storage, int n) {
    int jobs = 10;
    int T = 1000;

    if (col_major_storage) {
        big_matrix = Eigen::MatrixXd::Random(n, 1);
        weights = Eigen::MatrixXd ::Zero(n, 1);
    }
    else {
        big_matrix = Eigen::Matrix<double,
                                   Eigen::Dynamic,
                                   Eigen::Dynamic,
                                   Eigen::RowMajor>::Random(n, 1);
        weights = Eigen::Matrix<double,
                                Eigen::Dynamic,
                                Eigen::Dynamic,
                                Eigen::RowMajor>::Zero(n, 1);
    }

    for (int j = 1; j <= 10; j++) {
        cout << j << " thread(s): ";
        auto blocks = get_parallel_processing_blocks(j, n);
        Timer t;

        if (j == 1) {
            complete_elapse(blocks[0], T);
        }
        else {
            vector<thread> threads(j);
            for (int i = 0; i < j; i++) {
                threads[i] = thread(complete_elapse, blocks[i], T);
            }

            for (auto& thread : threads) {
                if (thread.joinable()) {
                    thread.join();
                }
            }
        }
    }
}

int main(int argc, char* argv[]) {
    eval_full_filter(true, 100000);
    //        eval_without_pool(true, 100000);
}