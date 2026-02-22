#pragma once

#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <stdexcept>

namespace selfplay {

// Thread-safe queue for game trajectories
template<typename T>
class ThreadSafeQueue {
public:
    void push(T value) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(std::move(value));
        cv_.notify_one();
    }

    bool try_pop(T& value) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty()) {
            return false;
        }
        value = std::move(queue_.front());
        queue_.pop();
        return true;
    }

    T pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return !queue_.empty() || done_; });

        if (queue_.empty()) {
            throw std::runtime_error("Queue is empty and done");
        }

        T value = std::move(queue_.front());
        queue_.pop();
        return value;
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }

    void set_done() {
        std::lock_guard<std::mutex> lock(mutex_);
        done_ = true;
        cv_.notify_all();
    }

private:
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::queue<T> queue_;
    std::atomic<bool> done_{false};
};

} // namespace selfplay
