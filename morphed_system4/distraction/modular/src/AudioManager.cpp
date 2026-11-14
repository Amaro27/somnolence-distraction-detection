#include "AudioManager.hpp"
#include "SoundPlayer.hpp"

AudioManager::AudioManager() {
    SoundPlayer::warmup();
}

AudioManager::~AudioManager() {
    stop();
}

void AudioManager::play(const std::string& path, float gain_db) {
    if (path.empty()) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    SoundPlayer::play_async(path, gain_db);
}

bool AudioManager::play_blocking(const std::string& path, float gain_db) {
    if (path.empty()) {
        return false;
    }
    // Intentionally do not hold mutex_ across play_sync so stop() can interrupt immediately.
    return SoundPlayer::play_sync(path, gain_db);
}

void AudioManager::stop() {
    SoundPlayer::stop();
}
