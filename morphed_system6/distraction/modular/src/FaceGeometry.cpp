#include "FaceGeometry.hpp"

#include <algorithm>

namespace FaceGeometry {

double compute_ear(const FacialLandmarks68& landmarks) noexcept {
    using namespace Index;
    auto eye_aspect_ratio = [&](size_t p1, size_t p2, size_t p3, size_t p4, size_t p5, size_t p6) noexcept {
        double vert_a = cv::norm(landmarks[p2] - landmarks[p6]);
        double vert_b = cv::norm(landmarks[p3] - landmarks[p5]);
        double horiz  = cv::norm(landmarks[p1] - landmarks[p4]);
        if (horiz < 1e-6) return 0.0;
        return (vert_a + vert_b) / (2.0 * horiz);
    };

    double ear_r = eye_aspect_ratio(RIGHT_EYE_P1, RIGHT_EYE_P2, RIGHT_EYE_P3, RIGHT_EYE_P4, RIGHT_EYE_P5, RIGHT_EYE_P6);
    double ear_l = eye_aspect_ratio(LEFT_EYE_P1,  LEFT_EYE_P2,  LEFT_EYE_P3,  LEFT_EYE_P4,  LEFT_EYE_P5,  LEFT_EYE_P6);
    return (ear_r + ear_l) * 0.5;
}

double compute_rho(const FacialLandmarks68& landmarks) noexcept {
    using namespace Index;
    const cv::Point2d p36(landmarks[RIGHT_EYE_P1].x, landmarks[RIGHT_EYE_P1].y);
    const cv::Point2d p45(landmarks[LEFT_EYE_P4].x, landmarks[LEFT_EYE_P4].y);

    const cv::Point2d e_mid = (p36 + p45) * 0.5;
    const cv::Point2d v_eyes = p45 - p36;
    const double w_eyes = cv::norm(v_eyes);

    cv::Point2d u_x;
    cv::Point2d u_y;
    if (w_eyes < 1e-6) {
        u_x = cv::Point2d(1.0, 0.0);
        u_y = cv::Point2d(0.0, 1.0);
    } else {
        u_x = v_eyes * (1.0 / w_eyes);
        u_y = cv::Point2d(-u_x.y, u_x.x); // Downward perpendicular
    }

    const cv::Point2d p33(landmarks[NOSE_TIP].x, landmarks[NOSE_TIP].y);
    const cv::Point2d p8(landmarks[CHIN].x, landmarks[CHIN].y);

    const double d_top = (p33 - e_mid).dot(u_y);
    const double d_bottom = (p8 - p33).dot(u_y);
    const double h_face = std::max(d_top + d_bottom, 1e-6);

    return d_top / h_face;
}

double compute_mar(const FacialLandmarks68& landmarks) noexcept {
    using namespace Index;
    double dist_a = cv::norm(landmarks[INNER_LIP_TOP_LEFT]  - landmarks[INNER_LIP_BOT_LEFT]);
    double dist_b = cv::norm(landmarks[INNER_LIP_TOP_MID]   - landmarks[INNER_LIP_BOT_MID]);
    double dist_c = cv::norm(landmarks[INNER_LIP_TOP_RIGHT] - landmarks[INNER_LIP_BOT_RIGHT]);
    double dist_d = cv::norm(landmarks[INNER_LIP_CORNER_LEFT] - landmarks[INNER_LIP_CORNER_RIGHT]);
    if (dist_d < 1e-6) return 0.0;
    return (dist_a + dist_b + dist_c) / (2.0 * dist_d);
}

} // namespace FaceGeometry
