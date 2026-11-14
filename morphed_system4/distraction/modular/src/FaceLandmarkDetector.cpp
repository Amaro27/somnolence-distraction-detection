#include "FaceLandmarkDetector.hpp"
#include "PathUtils.hpp"

#include <dlib/opencv.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>
#include <opencv2/imgproc.hpp>

FaceLandmarkDetector::FaceLandmarkDetector(FaceLandmarkConfig config)
    : config_(std::move(config)),
      face_lost_start_(std::chrono::steady_clock::now()) {}

std::string FaceLandmarkDetector::resolve_predictor_path() const {
    std::string resolved = PathUtils::resolve_asset_path(config_.predictor_path);
    std::error_code ec;
    if (!resolved.empty() && std::filesystem::exists(resolved, ec)) {
        return resolved;
    }
    return PathUtils::resolve_asset_path("input/shape_predictor_68_face_landmarks.dat");
}

bool FaceLandmarkDetector::initialize() {
    try {
        detector_ = dlib::get_frontal_face_detector();

        std::string resolved_path = resolve_predictor_path();
        std::ifstream test_f(resolved_path.c_str());
        if (!test_f.good()) {
            std::cerr << "[FaceLandmarkDetector] Error: Model file not accessible at: " << resolved_path << "\n";
            return false;
        }

        std::cout << "[FaceLandmarkDetector] Loading shape predictor from: " << resolved_path << " ...\n";
        dlib::deserialize(resolved_path) >> predictor_;
        initialized_ = true;
        std::cout << "[FaceLandmarkDetector] Initialized Dlib frontal face detector and 68-point shape predictor successfully.\n";
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[FaceLandmarkDetector] Fatal error during Dlib initialization: " << e.what() << "\n";
        return false;
    }
}

std::optional<FacialLandmarks68> FaceLandmarkDetector::detect(
    const cv::Mat& frame,
    std::chrono::steady_clock::time_point now
) {
    if (!initialized_) {
        return std::nullopt;
    }

    if (frame.empty()) {
        return std::nullopt;
    }

    // Convert to grayscale for Dlib HOG detector if 3-channel BGR
    if (frame.channels() == 3) {
        cv::cvtColor(frame, gray_buf_, cv::COLOR_BGR2GRAY);
    } else {
        gray_buf_ = frame;
    }

    dlib::cv_image<unsigned char> cimg(gray_buf_);

    // Run full frontal face detector on grayscale image
    std::vector<dlib::rectangle> faces = detector_(cimg, 0);

    if (faces.empty()) {
        if (!face_missing_) {
            face_missing_ = true;
            face_lost_start_ = now;
            warned_3s_ = false;
        } else {
            auto elapsed = std::chrono::duration<double>(now - face_lost_start_).count();
            if (elapsed >= config_.lost_face_timeout.count() && !warned_3s_) {
                std::cout << "\n[WARN] FaceLandmarkDetector: Driver face not found for "
                          << elapsed << "s (>=" << config_.lost_face_timeout.count()
                          << "s)! Searching entire frame...\n";
                warned_3s_ = true;
            }
        }
        return std::nullopt;
    }

    // Verified face detected: reset missing state
    face_missing_ = false;
    warned_3s_ = false;

    // Run 68-landmark shape predictor on primary detected face
    dlib::full_object_detection shape = predictor_(cimg, faces[0]);

    FacialLandmarks68 points;
    for (size_t i = 0; i < 68; ++i) {
        points[i] = cv::Point2f(
            static_cast<float>(shape.part(i).x()),
            static_cast<float>(shape.part(i).y())
        );
    }

    return points;
}

void FaceLandmarkDetector::reset() {
    face_missing_ = false;
    warned_3s_ = false;
    face_lost_start_ = std::chrono::steady_clock::now();
}
