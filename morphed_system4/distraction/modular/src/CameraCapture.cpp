#include "CameraCapture.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <iostream>
#include <utility>

CameraCapture::CameraCapture(std::string source, std::shared_ptr<ThreadSafeMailbox<cv::Mat>> mailbox)
    : source_(std::move(source)), mailboxes_{std::move(mailbox)} {}

CameraCapture::CameraCapture(std::string source, std::vector<std::shared_ptr<ThreadSafeMailbox<cv::Mat>>> mailboxes)
    : source_(std::move(source)), mailboxes_(std::move(mailboxes)) {}

CameraCapture::~CameraCapture() {
    stop();
}

bool CameraCapture::start() {
    if (is_running_.load()) {
        return true;
    }

    // Open capture device synchronously on calling thread (fail-fast single open)
    cv::VideoCapture cap;
    const bool is_numeric = !source_.empty() && std::all_of(source_.begin(), source_.end(), [](unsigned char c) {
        return std::isdigit(c);
    });

    if (is_numeric) {
        cap.open(std::stoi(source_));
    } else {
        cap.open(source_);
    }

    if (!cap.isOpened()) {
        std::cerr << "[!] CameraCapture Error: Failed to open video source '" << source_ << "'\n";
        return false;
    }

    is_running_.store(true);
    // Transfer opened handle directly into worker thread (eliminates double-open cycle)
    worker_thread_ = std::thread(&CameraCapture::capture_loop, this, std::move(cap), is_numeric);
    std::cout << "[+] CameraCapture thread started on source: " << source_ << "\n";
    return true;
}

void CameraCapture::stop() {
    is_running_.store(false);
    for (auto& mb : mailboxes_) {
        if (mb) {
            mb->stop();
        }
    }
    if (worker_thread_.joinable()) {
        worker_thread_.join();
        std::cout << "[*] CameraCapture thread stopped cleanly.\n";
    }
}

bool CameraCapture::is_running() const {
    return is_running_.load();
}

void CameraCapture::capture_loop(cv::VideoCapture cap, bool is_numeric) {
    double fps = cap.get(cv::CAP_PROP_FPS);
    int frame_delay_ms = 0;
    if (!is_numeric) {
        if (fps <= 1.0 || fps > 120.0) fps = 30.0;
        frame_delay_ms = static_cast<int>(1000.0 / fps);
    }

    cv::Mat frame;
    while (is_running_.load()) {
        auto frame_start = std::chrono::steady_clock::now();

        if (!cap.read(frame) || frame.empty()) {
            std::cout << "[*] CameraCapture: Stream reached end-of-file or frame grab failed.\n";
            break;
        }

        // cv::Mat uses shallow ref-counted copies, so deep clone per mailbox ensures thread isolation.
        for (auto& mb : mailboxes_) {
            if (mb) {
                mb->put(frame.clone());
            }
        }

        if (frame_delay_ms > 0) {
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - frame_start
            );
            if (elapsed.count() < frame_delay_ms) {
                std::this_thread::sleep_for(std::chrono::milliseconds(frame_delay_ms - elapsed.count()));
            }
        }
    }

    cap.release();
    is_running_.store(false);
    for (auto& mb : mailboxes_) {
        if (mb) {
            mb->stop();
        }
    }
}
