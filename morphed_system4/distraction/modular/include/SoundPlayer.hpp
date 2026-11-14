#pragma once

#include <string>

/**
 * @brief OS-independent audio playback functions.
 *
 * Uses native Win32 PlaySound on Windows, POSIX audio commands on Linux/macOS,
 * and no-op fallback on other platforms.
 */
namespace SoundPlayer {

bool play_async(const std::string& path);
bool play_sync(const std::string& path);
void stop();
std::string resolve_path(const std::string& path);

} // namespace SoundPlayer
