#include "../Tensor.h"
#include "Kernels.h"
#include <numeric>
#include <curand.h>

namespace ryupy
{
    std::shared_ptr<Tensor> Tensor::zeros(const std::vector<int> &shape)
    {
        std::shared_ptr<Tensor> tensor = std::make_shared<Tensor>(shape);

        int size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>()) * sizeof(float);

        int blockSize = 256;
        int numBlocks = (size + blockSize - 1) / blockSize;

        zerosKernel<<<numBlocks, blockSize>>>(tensor->d_data, size);

        return tensor;
    }

    std::shared_ptr<Tensor> Tensor::ones(const std::vector<int> &shape)
    {
        std::shared_ptr<Tensor> tensor = std::make_shared<Tensor>(shape);

        int size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>()) * sizeof(float);

        int blockSize = 256;
        int numBlocks = (size + blockSize - 1) / blockSize;

        onesKernel<<<numBlocks, blockSize>>>(tensor->d_data, size);

        return tensor;
    }

    std::shared_ptr<Tensor> Tensor::fill(const std::vector<int> &shape, float val)
    {
        std::shared_ptr<Tensor> tensor = std::make_shared<Tensor>(shape);

        int size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>()) * sizeof(float);

        int blockSize = 256;
        int numBlocks = (size + blockSize - 1) / blockSize;

        fillKernel<<<numBlocks, blockSize>>>(tensor->d_data, val, size);

        return tensor;
    }

    std::shared_ptr<Tensor> Tensor::arange(float start, float stop, float step)
    {
        int size = static_cast<int>((stop - start) / step);
        std::vector<int> shape = {size};

        std::shared_ptr<Tensor> tensor = std::make_shared<Tensor>(shape);

        int blockSize = 256;
        int numBlocks = (size + blockSize - 1) / blockSize;
        arangeKernel<<<numBlocks, blockSize>>>(tensor->d_data, start, step, size);
        return tensor;
    }

    std::shared_ptr<Tensor> Tensor::linspace(float start, float stop, int num)
    {
        std::vector<int> shape = {num};
        auto tensor = std::make_shared<Tensor>(shape);
        float step = (stop - start) / (num - 1);

        int blockSize = 256;
        int numBlocks = (num + blockSize - 1) / blockSize;
        linspaceKernel<<<numBlocks, blockSize>>>(tensor->d_data, start, step, num);
        return tensor;
    }

    std::shared_ptr<Tensor> Tensor::eye(int n)
    {
        std::vector<int> shape = {n, n};
        auto tensor = std::make_shared<Tensor>(shape);

        int blockSize = 256;
        int numBlocks = (n * n + blockSize - 1) / blockSize;
        eyeKernel<<<numBlocks, blockSize>>>(tensor->d_data, n);
        return tensor;
    }

    std::shared_ptr<Tensor> Tensor::random_uniform(const std::vector<int> &shape, float low, float high)
    {
        auto tensor = std::make_shared<Tensor>(shape);

        curandGenerator_t gen;
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);

        curandSetPseudoRandomGeneratorSeed(gen, static_cast<unsigned long long>(time(nullptr)));

        curandGenerateUniform(gen, tensor->d_data, tensor->size);

        if (low != 0.0f || high != 1.0f)
        {
            float scale = high - low;
            int blockSize = 256;
            int numBlocks = (tensor->size + blockSize - 1) / blockSize;
            scaleKernel<<<numBlocks, blockSize>>>(tensor->d_data, low, scale, tensor->size);
        }
        curandDestroyGenerator(gen);
        return tensor;
    }

    std::shared_ptr<Tensor> Tensor::random_normal(const std::vector<int> &shape, float mean, float std)
    {
        auto tensor = std::make_shared<Tensor>(shape);

        curandGenerator_t gen;
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);

        curandSetPseudoRandomGeneratorSeed(gen, static_cast<unsigned long long>(time(nullptr)));

        curandGenerateNormal(gen, tensor->d_data, tensor->size, mean, std);
        curandDestroyGenerator(gen);
        return tensor;
    }

    std::shared_ptr<Tensor> Tensor::xavier_uniform(const std::vector<int> &shape)
    {
        std::pair<int, int> fans = calculate_fans(shape);
        int fan_in = fans.first;
        int fan_out = fans.second;
        float bound = std::sqrt(6.0f / (fan_in + fan_out));
        return random_uniform(shape, -bound, bound);
    }

    std::shared_ptr<Tensor> Tensor::xavier_normal(const std::vector<int> &shape)
    {
        std::pair<int, int> fans = calculate_fans(shape);
        int fan_in = fans.first;
        int fan_out = fans.second;
        float std = std::sqrt(2.0f / (fan_in + fan_out));
        return random_normal(shape, 0.0f, std);
    }

    std::shared_ptr<Tensor> Tensor::kaiming_uniform(const std::vector<int> &shape)
    {
        std::pair<int, int> fans = calculate_fans(shape);
        int fan_in = fans.first;
        int fan_out = fans.second;
        float bound = std::sqrt(2.0f) * std::sqrt(3.0f / fan_in);
        return random_uniform(shape, -bound, bound);
    }

    std::shared_ptr<Tensor> Tensor::kaiming_normal(const std::vector<int> &shape)
    {
        std::pair<int, int> fans = calculate_fans(shape);
        int fan_in = fans.first;
        int fan_out = fans.second;
        float std = std::sqrt(2.0f / fan_in);
        return random_normal(shape, 0.0f, std);
    }
}