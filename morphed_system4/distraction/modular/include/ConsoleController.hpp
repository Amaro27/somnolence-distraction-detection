#pragma once

#include <atomic>
#include <cstdint>
#include <thread>

enum class ConsoleCommand : uint8_t {
    None = 0,
    ToggleMovement,
    Calibrate,
    Quit
};

/**
 * @brief Non-blocking console keyboard listener for interactive control.
 *
 * Controls:
 *   'm' / 'M' - Toggle vehicle movement state (Stationary <-> Moving).
 *   'c' / 'C' - Trigger physiological baseline calibration.
 *   'q' / 'Q' - Cleanly shutdown and exit application.
 */
class ConsoleController {
public:
    ConsoleController() = default;
    ~ConsoleController();

    // Non-copyable, non-movable
    ConsoleController(const ConsoleController&) = delete;
    ConsoleController& operator=(const ConsoleController&) = delete;

    /**
     * @brief Starts the background keyboard listener thread.
     */
    void start();

    /**
     * @brief Signals thread to stop and waits for join.
     */
    void stop();

    /**
     * @brief Atomically retrieves and clears any pending console command.
     */
    ConsoleCommand poll_command() noexcept {
        return pending_cmd_.exchange(ConsoleCommand::None, std::memory_order_acq_rel);
    }

    bool is_running() const noexcept { return is_running_.load(std::memory_order_relaxed); }

private:
    void listener_loop();

    std::atomic<bool> is_running_{false};
    std::atomic<ConsoleCommand> pending_cmd_{ConsoleCommand::None};
    std::thread worker_thread_;
};
