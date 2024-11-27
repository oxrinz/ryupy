#include "Net.h"

#include <string>
#include <unordered_map>
#include <memory>
#include <stdexcept>

namespace ryupy
{
    namespace nn
    {
        Net::Net(std::shared_ptr<LayerBank> layer_bank, ForwardFunction forward_fn)
        {
            m_layer_bank = layer_bank;
            m_forward_fn = forward_fn;
        }

        std::shared_ptr<Net> Net::create(std::shared_ptr<LayerBank> layer_bank, ForwardFunction forward_fn)
        {
            return std::make_shared<Net>(layer_bank, forward_fn);
        }

        std::shared_ptr<Tensor> Net::forward(std::shared_ptr<Tensor> input)
        {
            return m_forward_fn(input);
        }
    }
}