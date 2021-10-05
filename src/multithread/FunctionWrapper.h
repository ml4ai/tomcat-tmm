#pragma once

#include <memory>
#include <mutex>
#include <queue>

namespace multithread {

    /**
     * Represents a non-copyable function wrapper.
     * This implementation follows the implementation in the book C++
     * Concurrency in Action by Anthony Williams.
     */
    class FunctionWrapper {
      public:
        //------------------------------------------------------------------
        // Constructors & Destructor
        //------------------------------------------------------------------

        /**
         * Creates a function wrapper.
         *
         * @tparam F: Function type
         * @param f: function to be wrapped
         */
        template <typename F>
        FunctionWrapper(F&& f) : impl(new impl_type<F>(std::move(f))) {}

        ~FunctionWrapper() {}

        //------------------------------------------------------------------
        // Copy & Move constructors/assignments
        //------------------------------------------------------------------

        FunctionWrapper& operator=(FunctionWrapper&& other) {
            impl = std::move(other.impl);
            return *this;
        }

        FunctionWrapper(const FunctionWrapper&) = delete;
        FunctionWrapper(FunctionWrapper&) = delete;
        FunctionWrapper& operator=(const FunctionWrapper&) = delete;

        //------------------------------------------------------------------
        // Member functions
        //------------------------------------------------------------------

        /**
         * Calls the wrapped function.
         */
        void call() { impl->call(); }

      private:
        //------------------------------------------------------------------
        // Structs
        //------------------------------------------------------------------
        struct impl_base {
            virtual void call() = 0;
            virtual ~impl_base() {}
        };
        template <typename F> struct impl_type : impl_base {
            F f;
            impl_type(F&& f_) : f(std::move(f_)) {}
            void call() { f(); }
        };

        //------------------------------------------------------------------
        // Data members
        //------------------------------------------------------------------
        std::unique_ptr<impl_base> impl;
    };
} // namespace multithread
