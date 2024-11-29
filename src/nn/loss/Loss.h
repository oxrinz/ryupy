#pragma once

#include "../../tensors/Tensor.h"

namespace ryupy
{
    namespace nn
    {
        namespace loss
        {
            std::shared_ptr<Tensor> mse_loss(Tensor &predictions, Tensor &targets);
        }
    }
}