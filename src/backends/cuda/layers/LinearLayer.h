#pragma once

#include "../tensors/CudaTensor.h"

namespace ryupy
{
    namespace cuda
    {
        namespace nn
        {
            class LinearLayer
            {
            public:
                enum class InitType
                {
                    XAVIER_UNIFORM,
                    XAVIER_NORMAL,
                    KAIMING_UNIFORM,
                    KAIMING_NORMAL
                };

                CudaTensor weight;
                CudaTensor bias;

                LinearLayer(int in_features, int out_features, InitType init_type);
                std::shared_ptr<CudaTensor> forward(ryupy::cuda::CudaTensor tensor);
            };
        }
    }
}