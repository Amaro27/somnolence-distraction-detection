#include "TelegramNotifier.hpp"

#include <cstdlib>
#include <curl/curl.h>
#include <iostream>
#include <string>

namespace {
std::string escape_json(const std::string& input) {
    std::string out;
    out.reserve(input.size() + 16);
    for (char c : input) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:   out += c; break;
        }
    }
    return out;
}
} // namespace

void telegram_init() {
    curl_global_init(CURL_GLOBAL_ALL);
}

void telegram_cleanup() {
    curl_global_cleanup();
}

void send_telegram_alert(const std::string& message) {
    CURL* curl = curl_easy_init();
    if (!curl) {
        std::cerr << "[Telegram] Error initializing CURL easy handle." << std::endl;
        return;
    }

    // Retrieve credentials from environment variables if set, otherwise use default
    const char* env_token = std::getenv("TELEGRAM_BOT_TOKEN");
    const char* env_chat_id = std::getenv("TELEGRAM_CHAT_ID");

    std::string bot_token = (env_token && *env_token) ? env_token : "TOKEN";
    std::string chat_id   = (env_chat_id && *env_chat_id) ? env_chat_id : "ID";

    std::string url = "https://api.telegram.org/bot" + bot_token + "/sendMessage";
    std::string json_payload = "{\"chat_id\": \"" + chat_id + "\", \"text\": \"" + escape_json(message) + "\"}";

    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_payload.c_str());
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 3L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 5L);

    CURLcode res = curl_easy_perform(curl);
    if (res != CURLE_OK) {
        std::cerr << "[Telegram] Failed to send alert: " << curl_easy_strerror(res) << std::endl;
    } else {
        std::cout << "[Telegram] Notification dispatched successfully." << std::endl;
    }

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
}
