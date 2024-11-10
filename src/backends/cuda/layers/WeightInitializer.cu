#include <random>
#include "../tensors/CudaTensor.h"

#include "WeightInitializer.h"

namespace ryupy
{
    namespace cuda
    {
        namespace nn
        {

            // Initialize static members
            std::random_device WeightInitializer::rd;
            std::mt19937 WeightInitializer::gen(WeightInitializer::rd());

            CudaTensor WeightInitializer::xavier_uniform(int fan_in, int fan_out)
            {
                float bound = std::sqrt(6.0f / (fan_in + fan_out));
                std::vector<float> h_data(fan_in * fan_out);
                std::uniform_real_distribution<float> dist(-bound, bound);

                for (int i = 0; i < fan_in * fan_out; i++)
                {
                    h_data[i] = dist(gen);
                }

                CudaTensor tensor(fan_in * fan_out, std::vector<int>{fan_out, fan_in});
                cudaMemcpy(tensor.d_data, h_data.data(), fan_in * fan_out * sizeof(float), cudaMemcpyHostToDevice);
                return tensor;
            }

            CudaTensor WeightInitializer::xavier_normal(int fan_in, int fan_out)
            {
                float std = std::sqrt(2.0f / (fan_in + fan_out));
                std::vector<float> h_data(fan_in * fan_out);
                std::normal_distribution<float> dist(0.0f, std);

                for (int i = 0; i < fan_in * fan_out; i++)
                {
                    h_data[i] = dist(gen);
                }

                CudaTensor tensor(fan_in * fan_out, std::vector<int>{fan_out, fan_in});
                cudaMemcpy(tensor.d_data, h_data.data(), fan_in * fan_out * sizeof(float), cudaMemcpyHostToDevice);
                return tensor;
            }

            CudaTensor WeightInitializer::kaiming_uniform(int fan_in, int fan_out)
            {
                float bound = std::sqrt(2.0f) * std::sqrt(3.0f / fan_in);
                std::vector<float> h_data(fan_in * fan_out);
                std::uniform_real_distribution<float> dist(-bound, bound);

                for (int i = 0; i < fan_in * fan_out; i++)
                {
                    h_data[i] = dist(gen);
                }

                CudaTensor tensor(fan_in * fan_out, std::vector<int>{fan_out, fan_in});
                cudaMemcpy(tensor.d_data, h_data.data(), fan_in * fan_out * sizeof(float), cudaMemcpyHostToDevice);
                return tensor;
            }

            CudaTensor WeightInitializer::kaiming_normal(int fan_in, int fan_out)
            {
                float std = std::sqrt(2.0f / fan_in);
                std::vector<float> h_data(fan_in * fan_out);
                std::normal_distribution<float> dist(0.0f, std);

                for (int i = 0; i < fan_in * fan_out; i++)
                {
                    h_data[i] = dist(gen);
                }

                CudaTensor tensor(fan_in * fan_out, std::vector<int>{fan_out, fan_in});
                cudaMemcpy(tensor.d_data, h_data.data(), fan_in * fan_out * sizeof(float), cudaMemcpyHostToDevice);
                return tensor;
            }

        }
    }
} 