cmake_minimum_required(VERSION 3.10)
project(PruebaTelegram)

find_package(CURL REQUIRED)

add_executable(prueba_telegram main.cpp)
target_link_libraries(prueba_telegram PRIVATE CURL::libcurl)
