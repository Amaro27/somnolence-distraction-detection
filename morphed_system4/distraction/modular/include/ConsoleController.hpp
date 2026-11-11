#pragma once

#include <atomic>
#include <functional>
#include <thread>

/**
 * @brief Non-blocking console keyboard listener for interactive demonstrations.
 *
 * Controls:
 *   'm' / 'M' - Toggle vehicle movement state (Stationary <-> Moving).
 *   'c' / 'C' - Trigger calibration routine (placeholder for drowsiness baseline).
 *   'q' / 'Q' - Cleanly shutdown and exit application.
 */
class ConsoleController {
public:
    using Callback = std::function<void()>;

    /**
     * @brief Constructs console controller with user callbacks.
     */
    ConsoleController(
        Callback on_toggle_movement,
        Callback on_calibrate,
        Callback on_quit
    );

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

    bool is_running() const { return is_running_.load(); }

private:
    void listener_loop();

    Callback on_toggle_movement_;
    Callback on_calibrate_;
    Callback on_quit_;

    std::atomic<bool> is_running_{false};
    std::thread worker_thread_;
};
