#include "ThreadPool.h"

namespace multithread {

    using namespace std;

    //----------------------------------------------------------------------
    // Constructors & Destructor
    //----------------------------------------------------------------------
    ThreadPool::ThreadPool(unsigned int num_threads) : done(false) {
        // Do not use more than 90% of the available cores/processors
        num_threads =
            min((int)num_threads, (int)(0.9 * thread::hardware_concurrency()));
        try {
            for (int i = 0; i < num_threads; i++) {
                this->threads.push_back(
                    thread(&ThreadPool::start_worker_thread, this));
            }
        }
        catch (...) {
            this->done = true;
            throw;
        }
    }

    ThreadPool::~ThreadPool() {
        this->done = true;
        for (auto& thread : this->threads) {
            if (thread.joinable()) {
                thread.join();
            }
        }
    }

    //----------------------------------------------------------------------
    // Member functions
    //----------------------------------------------------------------------
    void ThreadPool::start_worker_thread() {
        while (!this->done) {
            if (shared_ptr<FunctionWrapper> task = this->work_queue.try_pop()) {
                task->call();
            }
            else {
                this_thread::yield();
            }
        }
    }

    template <typename FunctionType>
    future<typename result_of<FunctionType()>::type>
    ThreadPool::submit(FunctionType f) {
        typedef typename result_of<FunctionType()>::type result_type;

        packaged_task<result_type()> task(move(f));
        std::future<result_type> res(task.get_future());
        this->work_queue.push(std::move(task));
        return res;
    }

    int ThreadPool::size() const {
        return this->threads.size();
    }

} // namespace multithread
