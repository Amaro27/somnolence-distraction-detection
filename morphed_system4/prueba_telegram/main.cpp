#include <iostream>
#include <string>
#include <cstdlib>
#include <curl/curl.h>

void enviarAlertaTelegram(const std::string& mensaje) {
    CURL *curl;
    CURLcode res;
    
    // Obtiene las credenciales desde variables de entorno (o sustituye localmente para pruebas)
    const char* env_token = std::getenv("TELEGRAM_BOT_TOKEN");
    const char* env_chat_id = std::getenv("TELEGRAM_CHAT_ID");

    std::string bot_token = env_token ? env_token : "TU_BOT_TOKEN_AQUI"; 
    std::string chat_id = env_chat_id ? env_chat_id : "TU_CHAT_ID_AQUI"; 
    
    std::string url = "https://api.telegram.org/bot" + bot_token + "/sendMessage";
    
    std::string json_payload = "{\"chat_id\": \"" + chat_id + "\", \"text\": \"" + mensaje + "\"}";

    curl = curl_easy_init();
    if(curl) {
        struct curl_slist *headers = NULL;
        headers = curl_slist_append(headers, "Content-Type: application/json");
        
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_payload.c_str());
        
        res = curl_easy_perform(curl);
        
        if(res != CURLE_OK) {
            std::cerr << "Fallo al conectar con Telegram: " << curl_easy_strerror(res) << std::endl;
        } else {
            std::cout << "\nNotificacion despachada con exito." << std::endl;
        }
        
        curl_easy_cleanup(curl);
        curl_slist_free_all(headers);
    }
}

int main() {
    curl_global_init(CURL_GLOBAL_ALL);
    
    std::cout << "Modulo de notificaciones iniciado..." << std::endl;
    
    std::string alerta = "⚠️ ALERTA: Prueba de conexión desde C++ en Windows exitosa.";
    enviarAlertaTelegram(alerta);
    
    curl_global_cleanup();
    return 0;
}