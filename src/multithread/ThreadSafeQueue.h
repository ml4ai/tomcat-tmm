#pragma once

#include <memory>
#include <mutex>
#include <queue>

namespace multithread {

    /**
     * Represents a thread safe queue.
     * This implementation follows the implementation in the book C++
     * Concurrency in Action by Anthony Williams.
     */
    template <typename T> class ThreadSafeQueue {
      public:
        //------------------------------------------------------------------
        // Constructors & Destructor
        //------------------------------------------------------------------

        /**
         * Creates an empty queue.
         */
        ThreadSafeQueue() {}

        ~ThreadSafeQueue() {}

        //------------------------------------------------------------------
        // Copy & Move constructors/assignments
        //------------------------------------------------------------------

        ThreadSafeQueue(const ThreadSafeQueue&) = default;

        ThreadSafeQueue& operator=(const ThreadSafeQueue&) = default;

        ThreadSafeQueue(ThreadSafeQueue&&) = default;

        ThreadSafeQueue& operator=(ThreadSafeQueue&&) = default;

        //------------------------------------------------------------------
        // Member functions
        //------------------------------------------------------------------

        /**
         * Waits until there's an element to pop in the queue and retrieve that
         * element when available.
         *
         * @param value: reference to the element to be popped.
         */
        void wait_and_pop(T& value) {
            std::unique_lock<std::mutex> lk(this->mutex);
            this->data_cond.wait(lk,
                                 [this] { return !this->data_queue.empty(); });
            value = std::move(*this->data_queue.front());
            this->data_queue.pop();
        }

        /**
         * Pops an element if the queue is not empty.
         *
         * @param value: reference to the element to be popped.
         *
         * @return: True if element was popped, false, otherwise.
         */
        bool try_pop(T& value) {
            std::lock_guard<std::mutex> lk(this->mutex);
            if (this->data_queue.empty())
                return false;
            value = std::move(*this->data_queue.front());
            this->data_queue.pop();
        }

        /**
         * Waits until there's an element to pop in the queue and retrieve that
         * element when available.
         *
         * @return: pointer to the popped element.
         */
        std::shared_ptr<T> wait_and_pop() {
            std::unique_lock<std::mutex> lk(this->mutex);
            this->data_cond.wait(lk,
                                 [this] { return !this->data_queue.empty(); });
            std::shared_ptr<T> res = this->data_queue.front();
            this->data_queue.pop();
            return res;
        }

        /**
         * Pops an element if the queue is not empty.
         *
         * @return: pointer to the popped element or null if there's no element
         * to be popped.
         */
        std::shared_ptr<T> try_pop() {
            std::lock_guard<std::mutex> lk(this->mutex);
            if (this->data_queue.empty())
                return std::shared_ptr<T>();
            std::shared_ptr<T> res = this->data_queue.front();
            this->data_queue.pop();
            return res;
        }

        /**
         * Whether the queue is empty.
         *
         * @return True if the queue is empty.
         */
        bool empty() const {
            std::lock_guard<std::mutex> lk(this->mutex);
            return this->data_queue.empty();
        }

        /**
         * Pushes a new element into the queue.
         *
         * @param new_value: element to be inserted into the queue.
         */
        void push(T new_value) {
            std::shared_ptr<T> data(std::make_shared<T>(std::move(new_value)));
            std::lock_guard<std::mutex> lk(this->mutex);
            this->data_queue.push(data);
            this->data_cond.notify_one();
        }

      private:
        //------------------------------------------------------------------
        // Data members
        //------------------------------------------------------------------

        mutable std::mutex mutex;

        std::queue<std::shared_ptr<T>> data_queue;

        std::condition_variable data_cond;
    };
} // namespace multithread
