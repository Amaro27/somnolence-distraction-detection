#pragma once

#include <mutex>
#include <string>

/**
 * @brief Unified audio playback service.
 *
 * Handles non-blocking and blocking WAV playback with dBFS gain control
 * and immediate cancellation. Contains zero detector or alert domain logic.
 */
class AudioManager {
public:
    AudioManager();
    ~AudioManager();

    AudioManager(const AudioManager&) = delete;
    AudioManager& operator=(const AudioManager&) = delete;

    /**
     * @brief Plays a WAV file asynchronously (non-blocking).
     */
    void play(const std::string& path, float gain_db = 0.0f);

    /**
     * @brief Plays a WAV file synchronously (blocks until finished or stopped).
     * @return true if playback completed normally, false if interrupted/failed.
     */
    bool play_blocking(const std::string& path, float gain_db = 0.0f);

    /**
     * @brief Immediately halts any active audio playback.
     */
    void stop();

    /**
     * @brief Checks if audio is currently playing.
     */
    bool is_playing() const;

private:
    std::mutex mutex_;
};
