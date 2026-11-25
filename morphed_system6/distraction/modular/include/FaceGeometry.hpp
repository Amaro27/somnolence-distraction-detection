#pragma once

#include "Types.hpp"
#include <opencv2/core.hpp>

namespace FaceGeometry {

// Semantic Dlib 68-Landmark Indices (Compile-time constants)
namespace Index {
    // Right Eye (Subject Right: 36..41)
    constexpr size_t RIGHT_EYE_P1 = 36;
    constexpr size_t RIGHT_EYE_P2 = 37;
    constexpr size_t RIGHT_EYE_P3 = 38;
    constexpr size_t RIGHT_EYE_P4 = 39;
    constexpr size_t RIGHT_EYE_P5 = 40;
    constexpr size_t RIGHT_EYE_P6 = 41;

    // Left Eye (Subject Left: 42..47)
    constexpr size_t LEFT_EYE_P1  = 42;
    constexpr size_t LEFT_EYE_P2  = 43;
    constexpr size_t LEFT_EYE_P3  = 44;
    constexpr size_t LEFT_EYE_P4  = 45;
    constexpr size_t LEFT_EYE_P5  = 46;
    constexpr size_t LEFT_EYE_P6  = 47;

    // Nose & Chin
    constexpr size_t CHIN         = 8;
    constexpr size_t NOSE_TIP     = 33;

    // Inner Lip Contour (60..67)
    constexpr size_t INNER_LIP_CORNER_LEFT  = 60;
    constexpr size_t INNER_LIP_TOP_LEFT     = 61;
    constexpr size_t INNER_LIP_TOP_MID      = 62;
    constexpr size_t INNER_LIP_TOP_RIGHT    = 63;
    constexpr size_t INNER_LIP_CORNER_RIGHT = 64;
    constexpr size_t INNER_LIP_BOT_RIGHT    = 65;
    constexpr size_t INNER_LIP_BOT_MID      = 66;
    constexpr size_t INNER_LIP_BOT_LEFT     = 67;
} // namespace Index

// Pure mathematical geometric functions (Stateless, noexcept, zero allocations)
double compute_ear(const FacialLandmarks68& landmarks) noexcept;
double compute_rho(const FacialLandmarks68& landmarks) noexcept;
double compute_mar(const FacialLandmarks68& landmarks) noexcept;

} // namespace FaceGeometry
