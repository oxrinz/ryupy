#include <memory>
#include <iostream>
#include "../../tensors/Tensor.h"

namespace ryupy
{
    namespace nn
    {
        namespace loss
        {

            std::shared_ptr<Tensor> mse_loss(Tensor &predictions, Tensor &targets)
            {
                if (predictions.shape != targets.shape)
                {
                    throw std::invalid_argument("Shape mismatch between predictions and targets");
                }

                float scale = 1.0f / (predictions.size / sizeof(float));
                auto diff = predictions - targets;
                auto squared = (*diff) * (*diff);
                float sum = squared->sum();
                auto result = Tensor::fill({1}, sum * scale);

                if (predictions.requires_grad)
                {
                    result->requires_grad = true;
                    result->is_leaf = false;
                    result->prev = {std::make_shared<Tensor>(predictions), std::make_shared<Tensor>(targets)};
                    result->backward_fn = [result, &predictions, &targets, scale]()
                    {
                        if (!predictions.grad)
                        {
                            predictions.grad = Tensor::zeros(predictions.shape);
                        }

                        auto diff = predictions - targets;
                        auto scale_tensor = Tensor::fill(predictions.shape, 2.0f * scale);
                        auto grad = (*diff) * (*scale_tensor);

                        *predictions.grad += *grad;
                    };
                }

                return result;
            }
        }
    }
}