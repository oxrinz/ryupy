#pragma once

#include "../../tensors/Tensor.h"

namespace ryupy
{
    namespace nn
    {
        namespace loss
        {
            float mse_loss(Tensor &predictions, Tensor &targets);
        }
    }
}