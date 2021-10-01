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
#include "utils/ThreadSafeQueue.h"

using namespace std;
using namespace tomcat::model;
using namespace multithread;

class FunctionWrapper {
    struct impl_base {
        virtual void call() = 0;
        virtual ~impl_base() {}
    };
    std::unique_ptr<impl_base> impl;
    template <typename F> struct impl_type : impl_base {
        F f;
        impl_type(F&& f_) : f(std::move(f_)) {}
        void call() { f(); }
    };

  public:
    template <typename F>
    FunctionWrapper(F&& f) : impl(new impl_type<F>(std::move(f))) {}

    void call() { impl->call(); }

    FunctionWrapper(FunctionWrapper&& other) : impl(std::move(other.impl)) {}

    FunctionWrapper& operator=(FunctionWrapper&& other) {
        impl = std::move(other.impl);
        return *this;
    }

    FunctionWrapper(const FunctionWrapper&) = delete;
    FunctionWrapper(FunctionWrapper&) = delete;
    FunctionWrapper& operator=(const FunctionWrapper&) = delete;
};

class ThreadPool {
  public:
    ThreadPool(unsigned int num_threads) : done(false) {
        // Do not use more than 90% of the available cores/processors
        num_threads =
            min((int)num_threads, (int)(0.9 * thread::hardware_concurrency()));
        try {
            for (int i = 0; i < num_threads; i++) {
                this->threads.push_back(
                    thread(&ThreadPool::worker_thread, this));
            }
        }
        catch (...) {
            this->done = true;
            throw;
        }
    }

    ~ThreadPool() {
        this->done = true;
        for (auto& thread : this->threads) {
            if (thread.joinable()) {
                thread.join();
            }
        }
    }

    void worker_thread() {
        while (!done) {
            if (shared_ptr<FunctionWrapper> task = this->work_queue.try_pop()) {
                task->call();
            }
            else {
                this_thread::yield();
            }
        }
    }

    // This assumes the function has no parameters and the return type is
    // determined by the result_of function. IF the function submitted to the
    // pool has parameters, use std::bind.
    template <typename FunctionType>
    future<typename result_of<FunctionType()>::type> submit(FunctionType f) {
        typedef typename result_of<FunctionType()>::type result_type;

        packaged_task<result_type()> task(std::move(f));
        std::future<result_type> res(task.get_future());
        work_queue.push(std::move(task));
        return res;
    }

  protected:
    std::atomic_bool done;
    ThreadSafeQueue<FunctionWrapper> work_queue;
    vector<thread> threads;
};

vector<double> pdfs = {0.2, 0.1, 0.3, 0.15, 0.25};
gsl_ran_discrete_t* ptable;

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
void resample(const pair<int, int>& block, shared_ptr<gsl_rng>& gen) {
    int initial_row = block.first;
    int num_rows = block.second;

    auto patch = big_matrix.block(initial_row, 0, num_rows, 1);

    //    int k = weights.size();
    //    const double* parameters_ptr = weights.col(0).data();
    //    unsigned int* sample_ptr = new unsigned int[k];
    gsl_ran_discrete_t* ptable2 =
        gsl_ran_discrete_preproc(weights.rows(), weights.col(0).data());

    for (int i = 0; i < num_rows; i++) {
        //        gsl_ran_multinomial(gen.get(), k, 1, parameters_ptr,
        //        sample_ptr); int p = distance(sample_ptr, find(sample_ptr,
        //        sample_ptr + k, 1));
        int p = gsl_ran_discrete(gen.get(), ptable2);
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
                ptable = gsl_ran_discrete_preproc(weights.rows(),
                                                  weights.col(0).data());
                resample(blocks[0], gen);
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
                ptable = gsl_ran_discrete_preproc(weights.rows(),
                                                  weights.col(0).data());

                for (int i = 0; i < blocks.size(); i++) {
                    futures[i] =
                        pool.submit(bind(resample, blocks[i], gens[i]));
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