#pragma once

#include "Types.hpp"

#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>

/**
 * @brief High-performance frame preprocessor for GazeResNetLSTM.
 *
 * Implements functional parity with PyTorch pipeline:
 * 1. Color conversion: BGR -> RGB
 * 2. Spatial Resize: (224, 224)
 * 3. CLAHE Contrast Enhancement: clipLimit=2.5, tileGridSize=(8, 8)
 * 4. ImageNet Affine Normalization & NCHW packing:
 *    channel = (pixel / 255.0 - mean) / std
 */
class Preprocessor {
public:
    /**
     * @brief Constructs preprocessor with explicit 2D dimensions.
     * @param input_dimension Neural network input resolution (default: 224x224).
     * @param clahe_clip CLAHE clip limit (default: 2.5).
     * @param clahe_grid CLAHE tile grid size (default: 8x8).
     */
    explicit Preprocessor(
        ImageDimension2D input_dimension = ImageDimension2D(224, 224),
        float clahe_clip = 2.5f,
        cv::Size clahe_grid = cv::Size(8, 8)
    );

    /**
     * @brief Processes a raw BGR frame into a 4D NCHW float32 blob (1 x 3 x H x W).
     * @param bgr_frame Input raw camera frame (CV_8UC3 BGR).
     * @param out_blob Pre-allocated or output 4D cv::Mat (CV_32F).
     */
    void process(const cv::Mat& bgr_frame, cv::Mat& out_blob);

    ImageDimension2D dimension() const { return dimension_; }

private:
    void pack_normalized_nchw(const cv::Mat& rgb_clahe, cv::Mat& out_blob);

    ImageDimension2D dimension_;
    cv::Ptr<cv::CLAHE> clahe_;

    // Reusable intermediate frame buffers to avoid per-frame heap allocations
    cv::Mat rgb_frame_;
    cv::Mat resized_rgb_;
    cv::Mat gray_resized_;
    cv::Mat clahe_gray_;
    cv::Mat clahe_rgb_;
};
