#pragma once

#include <string>

/**
 * @brief OS-independent audio playback interface for calibration sounds.
 *
 * Uses native Win32 PlaySound on Windows, POSIX audio commands on Linux/macOS,
 * and no-op fallback on other platforms.
 */
class SoundPlayer {
public:
    /**
     * @brief Plays a WAV audio file asynchronously (non-blocking).
     * @param path Path to .wav audio file.
     * @return true if playback initiated successfully, false otherwise.
     */
    static bool play_async(const std::string& path);

    /**
     * @brief Plays a WAV audio file synchronously (blocks until playback finishes or is stopped).
     * @param path Path to .wav audio file.
     * @return true if playback completed successfully, false if stopped or failed.
     */
    static bool play_sync(const std::string& path);

    /**
     * @brief Immediately halts any active audio playback.
     */
    static void stop();

    /**
     * @brief Resolves relative candidate paths for sound files across build and project directories.
     * @param path Relative or absolute path to sound file.
     * @return Existing path if found, or original path.
     */
    static std::string resolve_path(const std::string& path);
};
