#include "CameraCapture.hpp"

#include <algorithm>
#include <cctype>
#include <iostream>
#include <opencv2/videoio.hpp>

CameraCapture::CameraCapture(std::string source, std::shared_ptr<ThreadSafeMailbox<cv::Mat>> mailbox)
    : source_(std::move(source)), mailbox_(std::move(mailbox)) {}

CameraCapture::~CameraCapture() {
    stop();
}

bool CameraCapture::start() {
    if (is_running_.load()) {
        return true;
    }

    // Probe whether source can be opened before launching thread
    cv::VideoCapture cap;
    bool is_numeric = !source_.empty() && std::all_of(source_.begin(), source_.end(), [](unsigned char c) {
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
    cap.release();

    is_running_.store(true);
    worker_thread_ = std::thread(&CameraCapture::capture_loop, this);
    std::cout << "[+] CameraCapture thread started on source: " << source_ << "\n";
    return true;
}

void CameraCapture::stop() {
    if (is_running_.exchange(false)) {
        if (mailbox_) {
            mailbox_->stop();
        }
        if (worker_thread_.joinable()) {
            worker_thread_.join();
        }
        std::cout << "[*] CameraCapture thread stopped cleanly.\n";
    }
}

bool CameraCapture::is_running() const {
    return is_running_.load();
}

void CameraCapture::capture_loop() {
    cv::VideoCapture cap;
    bool is_numeric = !source_.empty() && std::all_of(source_.begin(), source_.end(), [](unsigned char c) {
        return std::isdigit(c);
    });

    if (is_numeric) {
        cap.open(std::stoi(source_));
    } else {
        cap.open(source_);
    }

    if (!cap.isOpened()) {
        std::cerr << "[!] CameraCapture thread error: Video source failed to open.\n";
        is_running_.store(false);
        if (mailbox_) mailbox_->stop();
        return;
    }

    cv::Mat frame;
    while (is_running_.load()) {
        if (!cap.read(frame) || frame.empty()) {
            std::cout << "[*] CameraCapture: Stream reached end-of-file or frame grab failed.\n";
            break;
        }

        if (mailbox_) {
            mailbox_->put(frame.clone());
        }
    }

    cap.release();
    is_running_.store(false);
    if (mailbox_) {
        mailbox_->stop();
    }
}
