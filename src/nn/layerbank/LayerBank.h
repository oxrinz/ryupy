#pragma once

#include <string>
#include <unordered_map>
#include <memory>
#include "../layers/Layer.h"

namespace ryupy
{
    namespace nn
    {
        class LayerBank
        {
        public:
            LayerBank() = default;

            std::unordered_map<std::string, std::shared_ptr<Layer>> layers;

            void setLayer(const std::string &name, std::shared_ptr<Layer> layer);

            std::shared_ptr<Layer> getLayer(const std::string &name);
            bool hasLayer(const std::string &name) const;

            static std::shared_ptr<LayerBank> create();
        };
    };
}