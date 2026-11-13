#include "AudioManager.hpp"
#include "SoundPlayer.hpp"

#include <utility>

namespace {
struct ScopedFlagReset {
    std::atomic<bool>& flag;
    ~ScopedFlagReset() { flag.store(false); }
};
} // namespace

AudioManager::AudioManager(AuditoryAlertConfig alert_cfg)
    : alert_cfg_(std::move(alert_cfg)) {}

AudioManager::~AudioManager() {
    stop();
}

void AudioManager::play(const std::string& path) {
    if (path.empty()) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    SoundPlayer::play_async(path);
}

bool AudioManager::play_blocking(const std::string& path) {
    if (path.empty()) {
        return false;
    }
    blocking_active_.store(true);
    ScopedFlagReset reset_guard{blocking_active_};
    // Intentionally do not hold mutex_ across play_sync so stop() can interrupt immediately.
    return SoundPlayer::play_sync(path);
}

void AudioManager::stop() {
    SoundPlayer::stop();
}

void AudioManager::play_alert(RiskLevel risk) {
    if (blocking_active_.load() || risk == RiskLevel::None) {
        return;
    }

    const std::string& path = (risk == RiskLevel::High)
        ? alert_cfg_.high_risk_sound_path
        : alert_cfg_.medium_risk_sound_path;

    if (path.empty()) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    SoundPlayer::play_async(path);
}
