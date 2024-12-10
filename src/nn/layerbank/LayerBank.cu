#include <string>
#include <unordered_map>
#include <memory>
#include <stdexcept>
#include "LayerBank.h"
#include "../layers/Layer.h"
#include <pybind11/pybind11.h>

namespace ryupy
{
    namespace nn
    {
        void LayerBank::setLayer(const std::string &name, std::shared_ptr<Layer> layer)
        {
            layers[name] = layer;
        } 

        std::shared_ptr<Layer> LayerBank::getLayer(const std::string &name)
        {
            auto it = layers.find(name);
            if (it == layers.end())
            {
                throw std::runtime_error("Layer '" + name + "' not found");
            }
            return it->second;
        }

        bool LayerBank::hasLayer(const std::string &name) const
        {
            return layers.find(name) != layers.end();
        }

        std::shared_ptr<LayerBank> LayerBank::create()
        {
            return std::make_shared<LayerBank>();
        }
    };
}