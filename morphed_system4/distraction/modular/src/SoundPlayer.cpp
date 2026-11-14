#include "SoundPlayer.hpp"
#include "PathUtils.hpp"

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

namespace SoundPlayer {

std::string resolve_path(const std::string& path) {
    return PathUtils::resolve_asset_path(path);
}

bool play_async(const std::string& path) {
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

bool play_sync(const std::string& path) {
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

void stop() {
#if defined(_WIN32)
    ::PlaySoundA(NULL, NULL, 0);
#elif defined(__linux__) || defined(__APPLE__)
    std::system("killall -q aplay paplay > /dev/null 2>&1");
#endif
}

} // namespace SoundPlayer
