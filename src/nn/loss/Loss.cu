#include <memory>
#include <iostream>
#include "../../tensors/Tensor.h"

namespace ryupy
{
    namespace nn
    {
        namespace loss
        {

            float mse_loss(Tensor &predictions, Tensor &targets)
            {
                if (predictions.shape.size() != targets.shape.size())
                {
                    throw std::invalid_argument("Number of dimensions mismatch");
                }

                auto diff = predictions - targets;
                auto squared = (*diff) * (*diff);
                float scale = 1.0f / predictions.size;
                float sum = squared->sum();
                float result = sum * scale;
                return result;
            }
        } 
    } 
} 