#pragma once

// CPU-only preset selection shared by the CLI and service. A preset is a
// caller's declaration; the PRKT header verifies architecture, not provenance.
#include <cstdint>
#include <cerrno>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace paraketto::models {

struct Preset {
    const char* key;
    int architecture_version;
    const char* filename;
    const char* label;
    const char* download_url;
};

inline const Preset& preset(const std::string& key) {
    static constexpr Preset v2{
        "v2", 2, "paraketto-fp16.bin", "Parakeet V2",
        "https://huggingface.co/localoptima/paraketto/resolve/main/paraketto-fp16.bin"};
    static constexpr Preset v3{
        "v3", 3, "paraketto-v3-fp16.bin", "Parakeet V3",
        "https://huggingface.co/localoptima/paraketto/resolve/main/paraketto-v3-fp16.bin"};
    static constexpr Preset ultra{
        "ultra", 3, "paraketto-ultra-fp16.bin", "Parakeet Ultra", ""};
    if (key == "v2" || key == "2") return v2;
    if (key == "v3" || key == "3") return v3;
    if (key == "ultra") return ultra;
    throw std::invalid_argument("Unknown model: " + key + " (use v2, v3, or ultra)");
}

inline std::string weights_path(const Preset& selected,
                                const std::string& executable_directory,
                                const std::string& cache_directory,
                                const std::string& override_path = {}) {
    if (!override_path.empty()) return override_path;
    const auto beside = std::filesystem::path(executable_directory) / selected.filename;
    std::error_code error;
    if (!executable_directory.empty() && std::filesystem::is_regular_file(beside, error))
        return beside.string();
    return (std::filesystem::path(cache_directory) / selected.filename).string();
}

inline int read_architecture(const std::string& path) {
    const int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NONBLOCK);
    if (fd < 0)
        throw std::runtime_error("Cannot open model weights: " + path);
    struct CloseFile {
        int fd;
        ~CloseFile() { close(fd); }
    } close_file{fd};
    struct stat status{};
    if (fstat(fd, &status) != 0 || !S_ISREG(status.st_mode))
        throw std::runtime_error("Model weights are missing or not a regular file: " + path);
    unsigned char header[8]{};
    std::size_t received = 0;
    while (received < sizeof(header)) {
        const auto count = read(fd, header + received, sizeof(header) - received);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) throw std::runtime_error("Cannot read PRKT model header: " + path);
        received += static_cast<std::size_t>(count);
    }
    const auto little_u32 = [](const unsigned char* p) -> std::uint32_t {
        return std::uint32_t(p[0]) | (std::uint32_t(p[1]) << 8) |
               (std::uint32_t(p[2]) << 16) | (std::uint32_t(p[3]) << 24);
    };
    if (little_u32(header) != 0x544B5250u)
        throw std::runtime_error("Invalid PRKT model header: " + path);
    const auto version = little_u32(header + 4);
    if (version != 2 && version != 3)
        throw std::runtime_error("Unsupported PRKT model architecture " +
                                 std::to_string(version) + ": " + path);
    return static_cast<int>(version);
}

inline void validate_architecture(const Preset& selected, int actual) {
    if (selected.architecture_version != actual)
        throw std::runtime_error(std::string("Model ") + selected.key +
            " requires architecture v" + std::to_string(selected.architecture_version) +
            ", but weights have architecture v" + std::to_string(actual));
}

inline std::string label(const Preset* selected, int actual) {
    if (selected) return selected->label;
    if (actual == 2 || actual == 3)
        return "Parakeet V" + std::to_string(actual) + " (custom weights)";
    return "Parakeet (custom weights)";
}

} // namespace paraketto::models
