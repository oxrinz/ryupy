#pragma once

namespace ryupy
{
    namespace nn
    {
        namespace optim
        {
            class Optimizer
            {
            protected:
                float lr;
                std::shared_ptr<LayerBank> layer_bank;
                std::unordered_map<std::shared_ptr<Tensor>, std::unordered_map<std::string, std::shared_ptr<Tensor>>> state;

                std::vector<std::shared_ptr<Tensor>> get_parameters()
                {
                    std::vector<std::shared_ptr<Tensor>> params;
                    for (const auto &layer_pair : *layer_bank)
                    {
                        auto layer = layer_pair.second;
                        auto layer_params = layer->parameters();
                        params.insert(params.end(), layer_params.begin(), layer_params.end());
                    }
                    return params;
                }

            public:
                explicit Optimizer(std::shared_ptr<LayerBank> bank, float learning_rate = 0.01f);

                virtual ~Optimizer() = default;

                virtual void step() = 0;
            };

            class SGD : public Optimizer
            {
            private:
                float momentum;
                float dampering;
                float weight_decay;
                bool nasterov;

            public:
                SGD(std::shared_ptr<LayerBank> bank, float lr = 0.01, float momentum = 0.0, float dampering = 0.0, float weight_decay = 0.0, bool nasterov = false);

                void step() override;
            };
        }
    }
}