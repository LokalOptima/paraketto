#include "model_selection.h"

#include <array>
#include <cstdio>
#include <fstream>
#include <functional>
#include <vector>

namespace fs = std::filesystem;
using namespace paraketto::models;

static void require(bool condition, const std::string& reason) {
    if (!condition) throw std::runtime_error(reason);
}

static void rejects(const std::function<void()>& action, const std::string& reason) {
    try { action(); }
    catch (const std::exception&) { return; }
    throw std::runtime_error(reason);
}

struct TemporaryDirectory {
    fs::path path;
    TemporaryDirectory() {
        char pattern[] = "/tmp/paraketto-model-selection-XXXXXX";
        char* made = mkdtemp(pattern);
        if (!made) throw std::runtime_error("mkdtemp failed");
        path = made;
    }
    ~TemporaryDirectory() {
        std::error_code error;
        fs::remove_all(path, error);
    }
};

static void write_header(const fs::path& file, unsigned char version) {
    std::ofstream output(file, std::ios::binary);
    const std::array<unsigned char, 8> header{{'P', 'R', 'K', 'T', version, 0, 0, 0}};
    output.write(reinterpret_cast<const char*>(header.data()), header.size());
    if (!output) throw std::runtime_error("could not write fixture");
}

static void test_presets() {
    require(&preset("2") == &preset("v2"), "legacy V2 alias changed");
    require(&preset("3") == &preset("v3"), "legacy V3 alias changed");
    const auto& v2 = preset("v2");
    const auto& v3 = preset("v3");
    const auto& ultra = preset("ultra");
    require(v2.architecture_version == 2, "V2 architecture");
    require(v3.architecture_version == 3 && ultra.architecture_version == 3,
            "Ultra shares the V3 architecture");
    require(std::string(v2.filename) == "paraketto-fp16.bin", "V2 filename");
    require(std::string(v3.filename) == "paraketto-v3-fp16.bin", "V3 filename");
    require(std::string(ultra.filename) == "paraketto-ultra-fp16.bin", "Ultra filename");
    for (const auto* model : {&v2, &v3})
        require(std::string(model->download_url) ==
            "https://huggingface.co/localoptima/paraketto/resolve/main/" +
            std::string(model->filename), "published preset download URL");
    require(std::string(ultra.download_url).empty(), "Ultra must not invent a download URL");
    for (const std::string bad : {"", "v1", "4", "ULTRA", "custom"})
        rejects([&] { (void)preset(bad); }, "unknown model accepted: " + bad);
}

static void test_resolution(const fs::path& root) {
    const auto executable = root / "bin";
    const auto cache = root / "cache";
    fs::create_directory(executable);
    fs::create_directory(cache);
    for (const std::string key : {"v2", "v3", "ultra"}) {
        const auto& model = preset(key);
        const auto local_file = executable / model.filename;
        const auto cache_file = cache / model.filename;
        require(weights_path(model, executable, cache) == cache_file.string(),
                "missing preset should resolve to cache preparation target");
        write_header(cache_file, model.architecture_version);
        require(weights_path(model, executable, cache) == cache_file.string(),
                "existing cached preset should resolve");
        write_header(local_file, model.architecture_version);
        require(weights_path(model, executable, cache) == local_file.string(),
                "beside-executable preset should precede cache");
        const std::string explicit_path = "relative/custom file.bin";
        require(weights_path(model, executable, cache, explicit_path) == explicit_path,
                "explicit override must be preserved without existence checks");
        fs::remove(local_file);
        fs::create_directory(local_file);
        require(weights_path(model, executable, cache) == cache_file.string(),
                "directory must not shadow a valid cache file");
        fs::remove(local_file);
        fs::create_symlink(cache_file, local_file);
        require(weights_path(model, executable, cache) == local_file.string(),
                "symlink to prepared regular weights should resolve");
        require(weights_path(model, "", cache) == cache_file.string(),
                "empty executable directory must not search the working directory");
    }
}

static void test_headers(const fs::path& root) {
    const auto file = root / "header.bin";
    for (const int version : {2, 3}) {
        write_header(file, version);
        require(read_architecture(file) == version, "PRKT little-endian architecture detection");
        for (const std::string key : {"v2", "v3", "ultra"}) {
            const auto& selected = preset(key);
            if (selected.architecture_version == version) validate_architecture(selected, version);
            else rejects([&] { validate_architecture(selected, version); },
                         "explicit preset architecture mismatch accepted");
        }
    }
    write_header(file, 9);
    rejects([&] { (void)read_architecture(file); }, "unsupported version accepted");
    {
        std::ofstream output(file, std::ios::binary);
        const char wrong_magic[8] = {'T', 'K', 'R', 'P', 3, 0, 0, 0};
        output.write(wrong_magic, sizeof(wrong_magic));
    }
    rejects([&] { (void)read_architecture(file); }, "bad magic accepted");
    { std::ofstream output(file); output << "PRKT"; }
    rejects([&] { (void)read_architecture(file); }, "short header accepted");
    fs::remove(file);
    rejects([&] { (void)read_architecture(file); }, "missing file accepted");
    rejects([&] { (void)read_architecture(root); }, "directory accepted");
    rejects([&] { (void)read_architecture("/dev/null"); }, "device accepted");
    require(mkfifo(file.c_str(), 0600) == 0, "FIFO fixture creation");
    rejects([&] { (void)read_architecture(file); }, "FIFO accepted or blocked");
}

static void test_identity() {
    require(label(&preset("ultra"), 3) == "Parakeet Ultra", "declared Ultra label");
    require(label(&preset("v3"), 3) == "Parakeet V3", "declared V3 label");
    require(label(nullptr, 3) == "Parakeet V3 (custom weights)",
            "custom V3 must not be identified as Ultra");
    require(label(nullptr, 2) == "Parakeet V2 (custom weights)", "custom V2 label");
    require(label(nullptr, 0) == "Parakeet (custom weights)", "unavailable header label");
}

int main() {
    try {
        TemporaryDirectory temporary;
        test_presets();
        test_resolution(temporary.path);
        test_headers(temporary.path);
        test_identity();
        std::puts("model selection: preset, path, header, and identity tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "model selection test failed: %s\n", error.what());
        return 1;
    }
}
