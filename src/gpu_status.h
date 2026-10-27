#pragma once

#include <cuda_runtime.h>
#include <nvml.h>
#include <dlfcn.h>
#include <unistd.h>
#include <string>
#include <vector>

// NVML reports both device-wide memory and this process's allocation. Load it
// optionally so headless inference still works without the NVML shared library.
class GpuStatus {
    void* library_ = nullptr;
    nvmlDevice_t device_ = nullptr;
    decltype(&nvmlShutdown) shutdown_ = nullptr;
    decltype(&nvmlDeviceGetMemoryInfo) memory_ = nullptr;
    decltype(&nvmlDeviceGetComputeRunningProcesses) processes_ = nullptr;
public:
    GpuStatus() {
        library_ = dlopen("libnvidia-ml.so.1", RTLD_LAZY | RTLD_LOCAL);
        if (!library_) return;
        auto init = reinterpret_cast<decltype(&nvmlInit)>(dlsym(library_, "nvmlInit_v2"));
        auto handle = reinterpret_cast<decltype(&nvmlDeviceGetHandleByPciBusId)>(
            dlsym(library_, "nvmlDeviceGetHandleByPciBusId_v2"));
        if (!init || !handle || init() != NVML_SUCCESS) return;
        shutdown_ = reinterpret_cast<decltype(shutdown_)>(dlsym(library_, "nvmlShutdown"));
        memory_ = reinterpret_cast<decltype(memory_)>(dlsym(library_, "nvmlDeviceGetMemoryInfo"));
        processes_ = reinterpret_cast<decltype(processes_)>(dlsym(library_, "nvmlDeviceGetComputeRunningProcesses_v3"));
        char pci[32]{};
        if (cudaDeviceGetPCIBusId(pci, sizeof(pci), 0) == cudaSuccess)
            handle(pci, &device_);
    }
    ~GpuStatus() {
        if (shutdown_) shutdown_();
        if (library_) dlclose(library_);
    }
    std::string json() const {
        std::string out = "{\"status\":\"ready\",\"pid\":" + std::to_string(getpid());
        nvmlMemory_t memory{};
        constexpr double mib = 1024.0 * 1024.0;
        if (device_ && memory_ && memory_(device_, &memory) == NVML_SUCCESS) {
            out += ",\"vram_total_mb\":" + std::to_string(memory.total / mib);
            out += ",\"vram_used_mb\":" + std::to_string(memory.used / mib);
        }
        bool found = false;
        if (device_ && processes_) {
            unsigned int count = 16;
            std::vector<nvmlProcessInfo_t> processes(count);
            auto rc = processes_(device_, &count, processes.data());
            if (rc == NVML_ERROR_INSUFFICIENT_SIZE) {
                processes.resize(count);
                rc = processes_(device_, &count, processes.data());
            }
            if (rc == NVML_SUCCESS) {
                for (unsigned int i = 0; i < count; ++i) {
                    if (processes[i].pid == static_cast<unsigned>(getpid()) &&
                        processes[i].usedGpuMemory != NVML_VALUE_NOT_AVAILABLE) {
                        out += ",\"server_vram_mb\":" + std::to_string(processes[i].usedGpuMemory / mib);
                        found = true;
                        break;
                    }
                }
            }
        }
        if (!found) out += ",\"server_vram_mb\":null";
        return out + "}";
    }
};
