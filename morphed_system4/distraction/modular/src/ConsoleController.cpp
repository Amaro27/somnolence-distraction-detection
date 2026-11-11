#include "ConsoleController.hpp"

#include <chrono>
#include <iostream>

#if defined(_WIN32)
#include <conio.h>
#else
#include <termios.h>
#include <unistd.h>
#include <sys/select.h>

namespace {
bool posix_kbhit() {
    timeval tv{0, 0};
    fd_set fds;
    FD_ZERO(&fds);
    FD_SET(STDIN_FILENO, &fds);
    return select(STDIN_FILENO + 1, &fds, nullptr, nullptr, &tv) == 1;
}

char posix_getch() {
    char buf = 0;
    termios old_settings{};
    tcgetattr(STDIN_FILENO, &old_settings);
    termios new_settings = old_settings;
    new_settings.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &new_settings);
    if (read(STDIN_FILENO, &buf, 1) < 0) {
        buf = 0;
    }
    tcsetattr(STDIN_FILENO, TCSANOW, &old_settings);
    return buf;
}
} // namespace
#endif

ConsoleController::ConsoleController(
    Callback on_toggle_movement,
    Callback on_calibrate,
    Callback on_quit
) : on_toggle_movement_(std::move(on_toggle_movement)),
    on_calibrate_(std::move(on_calibrate)),
    on_quit_(std::move(on_quit)) {}

ConsoleController::~ConsoleController() {
    stop();
}

void ConsoleController::start() {
    if (is_running_.load()) return;

    is_running_.store(true);
    worker_thread_ = std::thread(&ConsoleController::listener_loop, this);
    std::cout << "[+] ConsoleController listener thread started.\n";
    std::cout << "    [Controls] 'm': Toggle vehicle movement (detect on/off)\n"
              << "               'c': Trigger calibration placeholder\n"
              << "               'q': Quit application cleanly\n";
}

void ConsoleController::stop() {
    if (is_running_.exchange(false)) {
        if (worker_thread_.joinable()) {
            worker_thread_.join();
        }
    }
}

void ConsoleController::listener_loop() {
    while (is_running_.load()) {
        bool hit = false;
        char ch = 0;

#if defined(_WIN32)
        if (_kbhit()) {
            ch = static_cast<char>(_getch());
            hit = true;
        }
#else
        if (posix_kbhit()) {
            ch = posix_getch();
            hit = true;
        }
#endif

        if (hit) {
            if (ch == 'm' || ch == 'M') {
                if (on_toggle_movement_) on_toggle_movement_();
            } else if (ch == 'c' || ch == 'C') {
                if (on_calibrate_) on_calibrate_();
            } else if (ch == 'q' || ch == 'Q') {
                if (on_quit_) on_quit_();
                break;
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}
