#pragma once

#include <random>
#include <cmath>

namespace ryupy
{
    namespace cuda
    {
        namespace nn
        {
            class WeightInitializer
            {
            private:
                static std::random_device rd;
                static std::mt19937 gen;

            public:
                static CudaTensor xavier_uniform(int fan_in, int fan_out);
                static CudaTensor xavier_normal(int fan_in, int fan_out);

                static CudaTensor kaiming_uniform(int fan_in, int fan_out);
                static CudaTensor kaiming_normal(int fan_int, int fan_out);
            };
        }
    }
}