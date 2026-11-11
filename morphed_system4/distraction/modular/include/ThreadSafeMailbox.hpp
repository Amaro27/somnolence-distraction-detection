#pragma once

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <utility>

/**
 * @brief Thread-safe single-slot mailbox with drop-oldest policy.
 *
 * Designed for real-time video streaming pipelines:
 * - Producer (camera capture thread) calls put(), which overwrites the latest item
 *   without blocking and notifies any waiting consumer.
 * - Consumer (inference pipeline) calls get_latest(), always receiving the freshest
 *   available frame without latency buildup or queue lag.
 */
template <typename T>
class ThreadSafeMailbox {
public:
    ThreadSafeMailbox() = default;

    // Non-copyable, non-movable to ensure synchronized thread safety
    ThreadSafeMailbox(const ThreadSafeMailbox&) = delete;
    ThreadSafeMailbox& operator=(const ThreadSafeMailbox&) = delete;

    /**
     * @brief Puts a new item into the mailbox, overwriting any previous unread item.
     * @param item The new item to store.
     */
    void put(T item) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            latest_item_ = std::move(item);
            has_new_ = true;
        }
        cv_.notify_one();
    }

    /**
     * @brief Waits for and retrieves the freshest item in the mailbox.
     * @param item Output destination for the retrieved item.
     * @param timeout Maximum duration to wait before returning false.
     * @return true if an item was successfully fetched, false if timed out or stopped.
     */
    bool get_latest(T& item, std::chrono::milliseconds timeout = std::chrono::milliseconds(100)) {
        std::unique_lock<std::mutex> lock(mutex_);
        bool received = cv_.wait_for(lock, timeout, [this] {
            return has_new_ || stopped_;
        });

        if (!received || (stopped_ && !has_new_)) {
            return false;
        }

        item = std::move(latest_item_);
        has_new_ = false;
        return true;
    }

    /**
     * @brief Signals shutdown to unblock any waiting consumer threads.
     */
    void stop() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopped_ = true;
        }
        cv_.notify_all();
    }

    /**
     * @brief Resets the mailbox state to allow reuse.
     */
    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        has_new_ = false;
        stopped_ = false;
    }

    /**
     * @brief Checks if a shutdown signal was dispatched.
     */
    bool is_stopped() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return stopped_;
    }

private:
    T latest_item_;
    bool has_new_{false};
    bool stopped_{false};
    mutable std::mutex mutex_;
    std::condition_variable cv_;
};
