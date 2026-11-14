#include "SoundPlayer.hpp"
#include "PathUtils.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <mutex>
#include <thread>
#include <unordered_map>
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
#elif defined(__linux__)
#include <csignal>
#include <cstdio>
#elif defined(__APPLE__)
#include <cstdlib>
#endif

namespace SoundPlayer {
namespace {

constexpr uint32_t TARGET_SAMPLE_RATE = 48000;
constexpr size_t TARGET_CHANNELS = 2;
constexpr size_t BLOCK_FRAMES = 480; // 10 ms at 48 kHz
constexpr size_t BLOCK_SAMPLES = BLOCK_FRAMES * TARGET_CHANNELS;
constexpr size_t NUM_RING_BUFFERS = 6; // 60 ms total ring buffer (6 x 10 ms)

uint16_t read_u16_le(const uint8_t* p) noexcept {
    return static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
}

uint32_t read_u32_le(const uint8_t* p) noexcept {
    return static_cast<uint32_t>(p[0]) |
           (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) |
           (static_cast<uint32_t>(p[3]) << 24);
}

float decode_sample_normalized(const uint8_t* ptr, uint16_t audio_format, uint16_t bits_per_sample) noexcept {
    if (audio_format == 1) {
        if (bits_per_sample == 8) {
            return (static_cast<float>(ptr[0]) - 128.0f) / 128.0f;
        }
        if (bits_per_sample == 16) {
            const int16_t val = static_cast<int16_t>(read_u16_le(ptr));
            return static_cast<float>(val) / 32768.0f;
        }
        if (bits_per_sample == 24) {
            int32_t val = static_cast<int32_t>(ptr[0]) |
                          (static_cast<int32_t>(ptr[1]) << 8) |
                          (static_cast<int32_t>(ptr[2]) << 16);
            if ((val & 0x00800000) != 0) {
                val |= static_cast<int32_t>(0xFF000000);
            }
            return static_cast<float>(val) / 8388608.0f;
        }
        if (bits_per_sample == 32) {
            const int32_t val = static_cast<int32_t>(read_u32_le(ptr));
            return static_cast<float>(static_cast<double>(val) / 2147483648.0);
        }
    } else if (audio_format == 3 && bits_per_sample == 32) {
        float val = 0.0f;
        std::memcpy(&val, ptr, sizeof(float));
        return val;
    }
    return 0.0f;
}

int16_t float_to_pcm16(float sample) noexcept {
    const float clamped = std::max(-1.0f, std::min(1.0f, sample));
    return static_cast<int16_t>(std::lround(clamped * 32767.0f));
}

std::vector<int16_t> decode_wav_to_48k_stereo(const std::string& resolved_path) {
    std::ifstream file(resolved_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        return {};
    }

    const std::streamsize file_size = file.tellg();
    if (file_size < 44) {
        return {};
    }

    std::vector<uint8_t> raw(static_cast<size_t>(file_size));
    file.seekg(0, std::ios::beg);
    if (!file.read(reinterpret_cast<char*>(raw.data()), file_size)) {
        return {};
    }

    if (std::memcmp(raw.data(), "RIFF", 4) != 0 || std::memcmp(raw.data() + 8, "WAVE", 4) != 0) {
        return {};
    }

    bool found_fmt = false;
    bool found_data = false;
    uint16_t audio_format = 0;
    uint16_t channels = 0;
    uint32_t sample_rate = 0;
    uint16_t block_align = 0;
    uint16_t bits_per_sample = 0;
    size_t data_offset = 0;
    size_t data_size = 0;

    size_t offset = 12;
    while (offset + 8 <= raw.size()) {
        const uint32_t chunk_size = read_u32_le(raw.data() + offset + 4);
        if (offset + 8u > raw.size()) {
            break;
        }
        const size_t available = raw.size() - (offset + 8u);
        const size_t safe_chunk_size = std::min(static_cast<size_t>(chunk_size), available);

        if (std::memcmp(raw.data() + offset, "fmt ", 4) == 0 && safe_chunk_size >= 16) {
            audio_format = read_u16_le(raw.data() + offset + 8);
            channels = read_u16_le(raw.data() + offset + 10);
            sample_rate = read_u32_le(raw.data() + offset + 12);
            block_align = read_u16_le(raw.data() + offset + 20);
            bits_per_sample = read_u16_le(raw.data() + offset + 22);
            if (audio_format == 0xFFFEu && safe_chunk_size >= 40) {
                audio_format = read_u16_le(raw.data() + offset + 32);
            }
            found_fmt = true;
        } else if (std::memcmp(raw.data() + offset, "data", 4) == 0) {
            data_offset = offset + 8u;
            data_size = safe_chunk_size;
            found_data = true;
            break;
        }

        const size_t padded_chunk_size = static_cast<size_t>(chunk_size) + (static_cast<size_t>(chunk_size) & 1u);
        if (offset + 8u + padded_chunk_size > raw.size()) {
            break;
        }
        offset += 8u + padded_chunk_size;
    }

    if (!found_fmt || !found_data || channels == 0 || sample_rate == 0 || bits_per_sample == 0) {
        return {};
    }
    if (!((audio_format == 1 && (bits_per_sample == 8 || bits_per_sample == 16 || bits_per_sample == 24 || bits_per_sample == 32)) ||
          (audio_format == 3 && bits_per_sample == 32))) {
        return {};
    }

    const size_t bytes_per_sample = static_cast<size_t>(bits_per_sample / 8u);
    const size_t frame_stride = (block_align > 0) ? static_cast<size_t>(block_align)
                                                  : static_cast<size_t>(channels) * bytes_per_sample;
    if (frame_stride == 0) {
        return {};
    }

    const size_t src_frames = data_size / frame_stride;
    if (src_frames == 0) {
        return {};
    }

    const uint8_t* pcm_bytes = raw.data() + data_offset;
    auto read_frame_lr = [&](size_t frame_idx, float& out_l, float& out_r) noexcept {
        const uint8_t* fptr = pcm_bytes + frame_idx * frame_stride;
        out_l = decode_sample_normalized(fptr, audio_format, bits_per_sample);
        out_r = (channels >= 2)
                    ? decode_sample_normalized(fptr + bytes_per_sample, audio_format, bits_per_sample)
                    : out_l;
    };

    std::vector<int16_t> out;
    if (sample_rate == TARGET_SAMPLE_RATE) {
        out.resize(src_frames * TARGET_CHANNELS);
        for (size_t i = 0; i < src_frames; ++i) {
            float l = 0.0f;
            float r = 0.0f;
            read_frame_lr(i, l, r);
            out[i * 2] = float_to_pcm16(l);
            out[i * 2 + 1] = float_to_pcm16(r);
        }
        return out;
    }

    const size_t dst_frames = static_cast<size_t>(
        std::ceil((static_cast<double>(src_frames) * TARGET_SAMPLE_RATE) / static_cast<double>(sample_rate))
    );
    out.resize(dst_frames * TARGET_CHANNELS);
    const double step = static_cast<double>(sample_rate) / static_cast<double>(TARGET_SAMPLE_RATE);

    for (size_t i = 0; i < dst_frames; ++i) {
        const double src_pos = static_cast<double>(i) * step;
        const size_t idx0 = std::min(static_cast<size_t>(src_pos), src_frames - 1);
        const size_t idx1 = std::min(idx0 + 1, src_frames - 1);
        const float frac = static_cast<float>(src_pos - static_cast<double>(idx0));

        float l0 = 0.0f, r0 = 0.0f;
        float l1 = 0.0f, r1 = 0.0f;
        read_frame_lr(idx0, l0, r0);
        read_frame_lr(idx1, l1, r1);

        out[i * 2] = float_to_pcm16(l0 + (l1 - l0) * frac);
        out[i * 2 + 1] = float_to_pcm16(r0 + (r1 - r0) * frac);
    }

    return out;
}

class AudioStreamEngine {
public:
    AudioStreamEngine() {
        start();
    }

    ~AudioStreamEngine() {
        shutdown();
    }

    void ensure_started() {
        // Singleton construction guarantees audio stream is initialized.
    }

    const std::vector<int16_t>* get_or_load_pcm(const std::string& resolved_path) {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        auto it = cache_.find(resolved_path);
        if (it != cache_.end()) {
            return it->second.empty() ? nullptr : &it->second;
        }

        std::vector<int16_t> pcm = decode_wav_to_48k_stereo(resolved_path);
        auto [inserted_it, inserted] = cache_.emplace(resolved_path, std::move(pcm));
        (void)inserted;
        return inserted_it->second.empty() ? nullptr : &inserted_it->second;
    }

    bool trigger(const std::string& resolved_path, bool wait_until_done) {
        const std::vector<int16_t>* samples = get_or_load_pcm(resolved_path);
        if (!samples || samples->empty()) {
            return false;
        }

        if (!running_.load(std::memory_order_acquire)) {
            return false;
        }

        uint64_t my_id = 0;
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            active_samples_ = samples;
            sample_cursor_ = 0;
            my_id = ++playback_id_;
        }

        if (!wait_until_done) {
            return true;
        }

        std::unique_lock<std::mutex> lock(state_mutex_);
        state_cv_.wait(lock, [&]() {
            return !running_.load(std::memory_order_acquire) ||
                   playback_id_ != my_id ||
                   active_samples_ == nullptr;
        });
        return playback_id_ == my_id && running_.load(std::memory_order_acquire);
    }

    void stop_playback() {
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            active_samples_ = nullptr;
            sample_cursor_ = 0;
            ++playback_id_;
        }
        state_cv_.notify_all();
    }

private:
    void fill_next_block(int16_t* dst) {
        bool finished = false;
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            if (active_samples_ != nullptr && sample_cursor_ < active_samples_->size()) {
                const size_t remaining = active_samples_->size() - sample_cursor_;
                const size_t to_copy = std::min(BLOCK_SAMPLES, remaining);
                std::memcpy(dst, active_samples_->data() + sample_cursor_, to_copy * sizeof(int16_t));
                if (to_copy < BLOCK_SAMPLES) {
                    std::memset(dst + to_copy, 0, (BLOCK_SAMPLES - to_copy) * sizeof(int16_t));
                }
                sample_cursor_ += to_copy;
                if (sample_cursor_ >= active_samples_->size()) {
                    active_samples_ = nullptr;
                    sample_cursor_ = 0;
                    finished = true;
                }
            } else {
                std::memset(dst, 0, BLOCK_SAMPLES * sizeof(int16_t));
            }
        }
        if (finished) {
            state_cv_.notify_all();
        }
    }

    void start() {
#if defined(_WIN32)
        WAVEFORMATEX wfx{};
        wfx.wFormatTag = WAVE_FORMAT_PCM;
        wfx.nChannels = static_cast<WORD>(TARGET_CHANNELS);
        wfx.nSamplesPerSec = TARGET_SAMPLE_RATE;
        wfx.wBitsPerSample = 16;
        wfx.nBlockAlign = static_cast<WORD>((wfx.nChannels * wfx.wBitsPerSample) / 8);
        wfx.nAvgBytesPerSec = wfx.nSamplesPerSec * wfx.nBlockAlign;

        wave_event_ = ::CreateEventA(nullptr, FALSE, FALSE, nullptr);
        if (!wave_event_) {
            return;
        }

        const MMRESULT res = ::waveOutOpen(
            &h_wave_out_,
            WAVE_MAPPER,
            &wfx,
            reinterpret_cast<DWORD_PTR>(wave_event_),
            0,
            CALLBACK_EVENT
        );
        if (res != MMSYSERR_NOERROR) {
            ::CloseHandle(wave_event_);
            wave_event_ = nullptr;
            h_wave_out_ = nullptr;
            return;
        }

        ring_pcm_.assign(NUM_RING_BUFFERS, std::vector<int16_t>(BLOCK_SAMPLES, 0));
        ring_headers_.assign(NUM_RING_BUFFERS, WAVEHDR{});
        for (size_t i = 0; i < NUM_RING_BUFFERS; ++i) {
            ring_headers_[i].lpData = reinterpret_cast<LPSTR>(ring_pcm_[i].data());
            ring_headers_[i].dwBufferLength = static_cast<DWORD>(BLOCK_SAMPLES * sizeof(int16_t));
            ::waveOutPrepareHeader(h_wave_out_, &ring_headers_[i], sizeof(WAVEHDR));
            ::waveOutWrite(h_wave_out_, &ring_headers_[i], sizeof(WAVEHDR));
        }

        running_.store(true, std::memory_order_release);
        mixer_thread_ = std::thread(&AudioStreamEngine::win32_mixer_loop, this);
#elif defined(__linux__)
        std::signal(SIGPIPE, SIG_IGN);
        running_.store(true, std::memory_order_release);
        mixer_thread_ = std::thread(&AudioStreamEngine::linux_mixer_loop, this);
#endif
    }

    void shutdown() {
        running_.store(false, std::memory_order_release);
        stop_playback();

#if defined(_WIN32)
        if (wave_event_) {
            ::SetEvent(wave_event_);
        }
#endif
        if (mixer_thread_.joinable()) {
            mixer_thread_.join();
        }

#if defined(_WIN32)
        if (h_wave_out_) {
            ::waveOutReset(h_wave_out_);
            for (auto& hdr : ring_headers_) {
                ::waveOutUnprepareHeader(h_wave_out_, &hdr, sizeof(WAVEHDR));
            }
            ::waveOutClose(h_wave_out_);
            h_wave_out_ = nullptr;
        }
        if (wave_event_) {
            ::CloseHandle(wave_event_);
            wave_event_ = nullptr;
        }
#endif
    }

#if defined(_WIN32)
    void win32_mixer_loop() {
        ::SetThreadPriority(::GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);
        size_t next_ring_idx = 0;

        while (running_.load(std::memory_order_acquire)) {
            bool wrote_any = false;
            for (size_t checked = 0; checked < NUM_RING_BUFFERS; ++checked) {
                WAVEHDR& hdr = ring_headers_[next_ring_idx];
                if ((hdr.dwFlags & WHDR_DONE) == 0) {
                    break;
                }
                fill_next_block(ring_pcm_[next_ring_idx].data());
                if (::waveOutWrite(h_wave_out_, &hdr, sizeof(WAVEHDR)) != MMSYSERR_NOERROR) {
                    break;
                }
                next_ring_idx = (next_ring_idx + 1) % NUM_RING_BUFFERS;
                wrote_any = true;
            }
            if (!wrote_any) {
                ::WaitForSingleObject(wave_event_, 15);
            }
        }
    }
#elif defined(__linux__)
    void linux_mixer_loop() {
        FILE* pipe = ::popen(
            "aplay -q -t raw -f S16_LE -c 2 -r 48000 --buffer-time=60000 - 2>/dev/null",
            "w"
        );
        if (!pipe) {
            return;
        }

        std::vector<int16_t> block(BLOCK_SAMPLES, 0);
        auto next_tick = std::chrono::steady_clock::now();

        while (running_.load(std::memory_order_acquire)) {
            fill_next_block(block.data());
            const size_t written = std::fwrite(block.data(), sizeof(int16_t), BLOCK_SAMPLES, pipe);
            if (written < BLOCK_SAMPLES || std::fflush(pipe) != 0) {
                break;
            }
            next_tick += std::chrono::milliseconds(10);
            const auto now = std::chrono::steady_clock::now();
            if (next_tick > now) {
                std::this_thread::sleep_until(next_tick);
            } else {
                next_tick = now;
            }
        }

        ::pclose(pipe);
    }
#endif

    std::mutex cache_mutex_;
    std::unordered_map<std::string, std::vector<int16_t>> cache_;

    std::mutex state_mutex_;
    std::condition_variable state_cv_;
    const std::vector<int16_t>* active_samples_{nullptr};
    size_t sample_cursor_{0};
    uint64_t playback_id_{0};

    std::atomic<bool> running_{false};
    std::thread mixer_thread_;

#if defined(_WIN32)
    HWAVEOUT h_wave_out_{nullptr};
    HANDLE wave_event_{nullptr};
    std::vector<std::vector<int16_t>> ring_pcm_;
    std::vector<WAVEHDR> ring_headers_;
#endif
};

AudioStreamEngine& stream_engine() {
    static AudioStreamEngine instance;
    return instance;
}

} // namespace

void warmup() {
    stream_engine().ensure_started();
    static constexpr const char* kDefaultSoundPaths[] = {
        "input/sounds/calib_phase_1.wav",
        "input/sounds/calib_phase_2.wav",
        "input/sounds/sine-415-Hz-1-second-padded.wav",
        "input/sounds/single-beep-350-hz.wav",
        "input/sounds/tada-LOUD.wav",
        "input/sounds/Ring06-LOUD.wav"
    };
    for (const char* raw_path : kDefaultSoundPaths) {
        const std::string resolved = resolve_path(raw_path);
        if (!resolved.empty()) {
            stream_engine().get_or_load_pcm(resolved);
        }
    }
}

std::string resolve_path(const std::string& path) {
    return PathUtils::resolve_asset_path(path);
}

bool play_async(const std::string& path) {
    std::string resolved = resolve_path(path);
    if (resolved.empty()) return false;

#if defined(_WIN32) || defined(__linux__)
    return stream_engine().trigger(resolved, false);
#elif defined(__APPLE__)
    std::string cmd = "afplay \"" + resolved + "\" > /dev/null 2>&1 &";
    return std::system(cmd.c_str()) == 0;
#else
    return false;
#endif
}

bool play_sync(const std::string& path) {
    std::string resolved = resolve_path(path);
    if (resolved.empty()) return false;

#if defined(_WIN32) || defined(__linux__)
    return stream_engine().trigger(resolved, true);
#elif defined(__APPLE__)
    std::string cmd = "afplay \"" + resolved + "\" > /dev/null 2>&1";
    return std::system(cmd.c_str()) == 0;
#else
    return false;
#endif
}

void stop() {
#if defined(_WIN32) || defined(__linux__)
    stream_engine().stop_playback();
#elif defined(__APPLE__)
    std::system("killall -q afplay > /dev/null 2>&1");
#endif
}

} // namespace SoundPlayer
