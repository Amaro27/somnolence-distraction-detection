#include "Preprocessor.hpp"

#include <cstdint>

namespace {
// ImageNet Channel Normalization Constants
constexpr float MEAN_R = 0.485f;
constexpr float MEAN_G = 0.456f;
constexpr float MEAN_B = 0.406f;

constexpr float STD_R = 0.229f;
constexpr float STD_G = 0.224f;
constexpr float STD_B = 0.225f;

// Precomputed affine multipliers: (pixel / 255.0f - mean) / std = pixel * SCALE + SHIFT
constexpr float SCALE_R = 1.0f / (255.0f * STD_R);
constexpr float SCALE_G = 1.0f / (255.0f * STD_G);
constexpr float SCALE_B = 1.0f / (255.0f * STD_B);

constexpr float SHIFT_R = -MEAN_R / STD_R;
constexpr float SHIFT_G = -MEAN_G / STD_G;
constexpr float SHIFT_B = -MEAN_B / STD_B;
} // namespace

Preprocessor::Preprocessor(cv::Size input_dimension, float clahe_clip, cv::Size clahe_grid)
    : dimension_(input_dimension),
      clahe_(cv::createCLAHE(clahe_clip, clahe_grid)),
      resized_rgb_(input_dimension.height, input_dimension.width, CV_8UC3),
      gray_resized_(input_dimension.height, input_dimension.width, CV_8UC1),
      clahe_gray_(input_dimension.height, input_dimension.width, CV_8UC1),
      clahe_rgb_(input_dimension.height, input_dimension.width, CV_8UC3) {}

void Preprocessor::process(const cv::Mat& bgr_frame, cv::Mat& out_blob) {
    // 1. Color convert: BGR -> RGB
    cv::cvtColor(bgr_frame, rgb_frame_, cv::COLOR_BGR2RGB);

    // 2. Spatial Resize: (width x height)
    cv::resize(rgb_frame_, resized_rgb_, cv::Size(dimension_.width, dimension_.height), 0.0, 0.0, cv::INTER_LINEAR);

    // 3. CLAHE Contrast Enhancement
    cv::cvtColor(resized_rgb_, gray_resized_, cv::COLOR_RGB2GRAY);
    clahe_->apply(gray_resized_, clahe_gray_);
    cv::cvtColor(clahe_gray_, clahe_rgb_, cv::COLOR_GRAY2RGB);

    // 4. Ensure destination blob is dimensioned (1 x 3 x H x W)
    const int blob_dims[4] = {1, 3, dimension_.height, dimension_.width};
    if (out_blob.dims != 4 || out_blob.size[0] != 1 || out_blob.size[1] != 3 ||
        out_blob.size[2] != dimension_.height || out_blob.size[3] != dimension_.width ||
        out_blob.type() != CV_32F) {
        out_blob.create(4, blob_dims, CV_32F);
    }

    // 5. Pack normalized NCHW planar blob
    pack_normalized_nchw(clahe_rgb_, out_blob);
}

void Preprocessor::pack_normalized_nchw(const cv::Mat& rgb_clahe, cv::Mat& out_blob) {
    const int height = rgb_clahe.rows;
    const int width = rgb_clahe.cols;
    const int hw = height * width;

    float* c0 = out_blob.ptr<float>(0, 0); // Red channel plane
    float* c1 = out_blob.ptr<float>(0, 1); // Green channel plane
    float* c2 = out_blob.ptr<float>(0, 2); // Blue channel plane

    if (rgb_clahe.isContinuous()) {
        const uint8_t* src = rgb_clahe.ptr<uint8_t>(0);
        for (int i = 0; i < hw; ++i) {
            c0[i] = static_cast<float>(src[i * 3 + 0]) * SCALE_R + SHIFT_R;
            c1[i] = static_cast<float>(src[i * 3 + 1]) * SCALE_G + SHIFT_G;
            c2[i] = static_cast<float>(src[i * 3 + 2]) * SCALE_B + SHIFT_B;
        }
    } else {
        int idx = 0;
        for (int y = 0; y < height; ++y) {
            const uint8_t* row = rgb_clahe.ptr<uint8_t>(y);
            for (int x = 0; x < width; ++x, ++idx) {
                c0[idx] = static_cast<float>(row[x * 3 + 0]) * SCALE_R + SHIFT_R;
                c1[idx] = static_cast<float>(row[x * 3 + 1]) * SCALE_G + SHIFT_G;
                c2[idx] = static_cast<float>(row[x * 3 + 2]) * SCALE_B + SHIFT_B;
            }
        }
    }
}
