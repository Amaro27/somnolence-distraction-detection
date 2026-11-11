#pragma once

#include "ThreadSafeMailbox.hpp"

#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <opencv2/core/mat.hpp>

/**
 * @brief Dedicated background worker thread that captures video frames.
 *
 * Runs camera capture decoupled from model inference and alarm evaluation.
 * Continuously pushes fresh frames into the ThreadSafeMailbox.
 */
class CameraCapture {
public:
    /**
     * @brief Constructs camera capture worker.
     * @param source Webcam index ("0") or path to a video file.
     * @param mailbox Shared mailbox receiving fresh frames.
     */
    CameraCapture(std::string source, std::shared_ptr<ThreadSafeMailbox<cv::Mat>> mailbox);

    /**
     * @brief Destructor ensures clean worker thread shutdown.
     */
    ~CameraCapture();

    // Non-copyable, non-movable
    CameraCapture(const CameraCapture&) = delete;
    CameraCapture& operator=(const CameraCapture&) = delete;

    /**
     * @brief Initializes capture device and starts worker thread.
     * @return true if capture stream opened successfully, false otherwise.
     */
    bool start();

    /**
     * @brief Signals thread to stop and waits for join.
     */
    void stop();

    /**
     * @brief Checks if capture thread is actively running.
     */
    bool is_running() const;

private:
    void capture_loop();

    std::string source_;
    std::shared_ptr<ThreadSafeMailbox<cv::Mat>> mailbox_;
    std::atomic<bool> is_running_{false};
    std::thread worker_thread_;
};
