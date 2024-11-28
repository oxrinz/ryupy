#pragma once
#include <string>
#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <sstream>

namespace ryupy
{
    struct GpuInfo
    {
        std::string name;
        size_t memory_total;
        int compute_capability_major;
        int compute_capability_minor;
        int max_threads_per_block;
        int warp_size;
    };

    struct SystemInfo
    {
        std::string ryupy_version = "0.1.0";
        std::string cuda_version;
        std::vector<GpuInfo> gpus;

        static SystemInfo get_info()
        {
            SystemInfo info;
            int cuda_version;
            cudaRuntimeGetVersion(&cuda_version);
            info.cuda_version = std::to_string(cuda_version / 1000) + "." +
                                std::to_string((cuda_version % 1000) / 10);

            int device_count;
            cudaGetDeviceCount(&device_count);

            for (int i = 0; i < device_count; i++)
            {
                cudaDeviceProp prop;
                cudaGetDeviceProperties(&prop, i);

                GpuInfo gpu;
                gpu.name = prop.name;
                gpu.memory_total = prop.totalGlobalMem / (1024 * 1024 * 1024);
                gpu.compute_capability_major = prop.major;
                gpu.compute_capability_minor = prop.minor;
                gpu.max_threads_per_block = prop.maxThreadsPerBlock;
                gpu.warp_size = prop.warpSize;

                info.gpus.push_back(gpu);
            }
            return info;
        }
    };

    inline void print_ryu()
    {
        const char *RED = "\033[31m";
        const char *WHITE = "\033[37m";
        const char *RESET = "\033[0m";

        SystemInfo sys_info = SystemInfo::get_info();
        const auto &gpu = sys_info.gpus[0]; // Get first GPU for display

        std::cout << RED << R"(                                                                                               
                █                          
       █         ███ █         █           
    ██       ██████████         █          
   █       ██████████████        █         
 ██      ███████ ██████  █       ██        
 █      ██████ ██ █████████      ██  █     
██     ██████    █  ███   ████    ██ ██    
██      █████          █          ███ ██    
██      █████           ████     ████  ██   
██      ██████                  █████ ███   
███      ██████               ██████  ███)"
                  << WHITE << "         RyuPy v" << sys_info.ryupy_version << RED << R"(   
███      ██████████     ███████████  ███)"
                  << WHITE << "          CUDA " << sys_info.cuda_version << RED << R"(
████       ███████████████████████  ████)"
                  << WHITE << "          " << gpu.name << " (" << gpu.memory_total << "GB)" << RED << R"(
████       ████████████████████   ████
██████       ████████████████████████ 
 ███████        ████████████████████       
  ████████████  ███████████████████        
   ██████████████████████████████          
    ███████████████████████████            
      █████████████████████                
         ██████                          
            ████████████                   
)" << RESET << std::endl;
    }

} // namespace ryupy