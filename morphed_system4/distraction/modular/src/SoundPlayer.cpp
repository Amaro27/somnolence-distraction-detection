#include "SoundPlayer.hpp"

#include <filesystem>
#include <iostream>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <mmsystem.h>
#elif defined(__linux__) || defined(__APPLE__)
#include <cstdlib>
#endif

std::string SoundPlayer::resolve_path(const std::string& path) {
    if (path.empty()) return "";
    std::error_code ec;
    if (std::filesystem::exists(path, ec)) {
        return path;
    }

    const std::vector<std::string> prefixes = {
        "morphed_system4/distraction/modular/",
        "input/sounds/",
        "morphed_system4/distraction/modular/input/sounds/",
        "../",
        "../../",
        "../../../",
        "../../../input/sounds/",
        "../../../morphed_system4/distraction/modular/input/sounds/"
    };

    for (const auto& prefix : prefixes) {
        std::string cand = prefix + path;
        if (std::filesystem::exists(cand, ec)) {
            return cand;
        }
    }
    return path;
}

bool SoundPlayer::play_async(const std::string& path) {
    std::string resolved = resolve_path(path);
    if (resolved.empty()) return false;

#if defined(_WIN32)
    return ::PlaySoundA(resolved.c_str(), NULL, SND_FILENAME | SND_ASYNC) != FALSE;
#elif defined(__linux__) || defined(__APPLE__)
    std::string cmd = "aplay -q \"" + resolved + "\" > /dev/null 2>&1 &";
    return std::system(cmd.c_str()) == 0;
#else
    return false;
#endif
}

bool SoundPlayer::play_sync(const std::string& path) {
    std::string resolved = resolve_path(path);
    if (resolved.empty()) return false;

#if defined(_WIN32)
    return ::PlaySoundA(resolved.c_str(), NULL, SND_FILENAME | SND_SYNC) != FALSE;
#elif defined(__linux__) || defined(__APPLE__)
    std::string cmd = "aplay -q \"" + resolved + "\" > /dev/null 2>&1";
    return std::system(cmd.c_str()) == 0;
#else
    return false;
#endif
}

void SoundPlayer::stop() {
#if defined(_WIN32)
    ::PlaySoundA(NULL, NULL, 0);
#elif defined(__linux__) || defined(__APPLE__)
    std::system("killall -q aplay paplay > /dev/null 2>&1");
#endif
}
