#include "../tensors/CudaTensor.h"
#include "LinearLayer.h"
#include "WeightInitializer.h"

namespace ryupy
{
    namespace cuda
    {
        namespace nn
        {
            LinearLayer::LinearLayer(int in_features, int out_features, InitType init_type)
            {
                switch (init_type)
                {
                case InitType::XAVIER_UNIFORM:
                    weight = WeightInitializer::xavier_uniform(in_features, out_features);
                    break;
                case InitType::XAVIER_NORMAL:
                    weight = WeightInitializer::xavier_normal(in_features, out_features);
                    break;
                case InitType::KAIMING_UNIFORM:
                    weight = WeightInitializer::kaiming_uniform(in_features, out_features);
                    break;
                case InitType::KAIMING_NORMAL:
                    weight = WeightInitializer::kaiming_normal(in_features, out_features);
                    break;
                }
            }

            std::shared_ptr<CudaTensor> LinearLayer::forward(ryupy::cuda::CudaTensor tensor)
            {
                std::shared_ptr<CudaTensor> output = tensor.matmul(weight);

                return output->operator+(bias);
            }
        }
    }
}