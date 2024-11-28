#pragma once

namespace ryupy
{
    namespace nn
    {
        namespace optim
        {

            SGD::SGD(std::shared_ptr<LayerBank> bank,
                     float lr = 0.01f,
                     float momentum = 0.0f,
                     float dampening = 0.0f,
                     float weight_decay = 0.0f,
                     bool nesterov = false)
                : Optimizer(bank, lr),
                  momentum(momentum),
                  dampening(dampening),
                  weight_decay(weight_decay),
                  nesterov(nesterov) {};

            void step()
            {
                auto parameters = get_parameters();
                for (auto &param : parameters)
                {
                    if (!param->grad)
                        continue;

                    auto d_p = param->grad->copy();

                    if (weight_decay != 0)
                    {
                        *d_p += (*param * weight_decay);
                    }

                    if (momentum != 0)
                    {
                        init_state(param);
                        auto &momentum_buffer = state[param]["momentum_buffer"];

                        *momentum_buffer = (*momentum_buffer * momentum) +
                                           (*d_p * (1.0f - dampening));

                        if (nesterov)
                        {
                            *d_p += (*momentum_buffer * momentum);
                        }
                        else
                        {
                            d_p = momentum_buffer;
                        }
                    }

                    *param -= (*d_p * lr);
                }
            }
        }
    }
}