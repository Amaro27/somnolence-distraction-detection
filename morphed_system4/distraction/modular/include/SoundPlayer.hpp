#pragma once

#include <string>

/**
 * @brief OS-independent audio playback functions.
 *
 * Uses native Win32 PlaySound on Windows, POSIX audio commands on Linux/macOS,
 * and no-op fallback on other platforms.
 */
namespace SoundPlayer {

void warmup();
bool play_async(const std::string& path, float gain_db = 0.0f);
bool play_sync(const std::string& path, float gain_db = 0.0f);
void stop();
std::string resolve_path(const std::string& path);

} // namespace SoundPlayer
