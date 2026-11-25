#pragma once

#include <string>

/**
 * @brief Initializes global CURL resources.
 * Call once during application startup (e.g. in main.cpp).
 */
void telegram_init();

/**
 * @brief Cleans up global CURL resources.
 * Call once during application shutdown.
 */
void telegram_cleanup();

/**
 * @brief Dispatches a standalone text alert message to the configured Telegram bot and chat.
 *
 * Reads credentials from environment variables:
 * - TELEGRAM_BOT_TOKEN
 * - TELEGRAM_CHAT_ID
 *
 * @param message The alert text to send.
 */
void send_telegram_alert(const std::string& message);

/**
 * @brief CamelCase convenience alias for send_telegram_alert.
 */
inline void sendTelegramAlert(const std::string& message) {
    send_telegram_alert(message);
}
