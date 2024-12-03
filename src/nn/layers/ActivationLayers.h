#pragma once
#include "../../tensors/Tensor.h"
#include <cudnn.h>

namespace ryupy
{
    namespace nn
    {
        std::shared_ptr<Tensor> relu(Tensor &input);
        std::shared_ptr<Tensor> sigmoid(Tensor &input);
        std::shared_ptr<Tensor> tanh(Tensor &input);
        std::shared_ptr<Tensor> leaky_relu(Tensor &input, float negative_slope = 0.01f);
    } // namespace nn
} // namespace ryupy