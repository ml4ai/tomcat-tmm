#pragma once

#include <future>
#include <thread>
#include <vector>

#include "multithread/ThreadSafeQueue.h"
#include "multithread/FunctionWrapper.h"

namespace multithread {

    /**
     * Represents a thread pool with a central task queue that are executed
     * by the threads in the pool.
     * This implementation follows the implementation in the book C++
     * Concurrency in Action by Anthony Williams.
     */
    class ThreadPool {
      public:
        //------------------------------------------------------------------
        // Constructors & Destructor
        //------------------------------------------------------------------

        /**
         * Creates a pool of threads.
         */
        ThreadPool(unsigned int num_threads);

        ~ThreadPool();

        //------------------------------------------------------------------
        // Member functions
        //------------------------------------------------------------------

        /**
         * Adds a task to the work queue to be performed by one of the threads
         * in the pool. The function submitted to the pool must have no
         * parameters. If the function has parameters, use std::bind before
         * submitting to the pool.
         *
         * @tparam FunctionType: Function type
         * @param f: function that contains the task to be executed
         * @return: function return
         */
        template <typename FunctionType>
        std::future<typename std::result_of<FunctionType()>::type>
        submit(FunctionType f);

        /**
         * Retrieves the number of threads in the pool.
         *
         * @return: Number of threads in the pool.
         */
        int size() const;

      protected:
        //------------------------------------------------------------------
        // Member functions
        //------------------------------------------------------------------

        /**
         * Starts a worker thread that runs until the destruction of the
         * pool.
         */
        void start_worker_thread();

        //------------------------------------------------------------------
        // Data members
        //------------------------------------------------------------------
        std::atomic_bool done;
        ThreadSafeQueue<FunctionWrapper> work_queue;
        std::vector<std::thread> threads;
    };

} // namespace multithread
