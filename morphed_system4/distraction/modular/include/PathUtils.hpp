#pragma once

#include <filesystem>
#include <initializer_list>
#include <string>
#include <string_view>

namespace PathUtils {

inline std::string resolve_asset_path(
    std::string_view relative_path,
    std::initializer_list<std::string_view> search_prefixes = {
        "",
        "morphed_system4/distraction/modular/",
        "input/",
        "input/sounds/",
        "morphed_system4/distraction/modular/input/",
        "morphed_system4/distraction/modular/input/sounds/",
        "../",
        "../../",
        "../../../",
        "../../../input/",
        "../../../input/sounds/",
        "../../../morphed_system4/distraction/modular/input/",
        "../../../morphed_system4/distraction/modular/input/sounds/"
    }
) {
    if (relative_path.empty()) return {};

    std::error_code ec;
    if (std::filesystem::exists(relative_path, ec)) {
        return std::string(relative_path);
    }

    for (const auto prefix : search_prefixes) {
        std::string candidate = (std::filesystem::path(prefix) / relative_path).string();
        if (std::filesystem::exists(candidate, ec)) {
            return candidate;
        }
    }

    return std::string(relative_path);
}

} // namespace PathUtils
