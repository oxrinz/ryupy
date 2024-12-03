#include "Dataset.h"
#include <random>
#include <algorithm>

namespace ryupy
{

    Dataset::Dataset(const std::map<std::string, std::vector<std::shared_ptr<Tensor>>> &inputs,
                     const std::map<std::string, std::vector<std::shared_ptr<Tensor>>> &targets)
        : input_tensors(inputs), target_tensors(targets)
    {
        reset();
    }

    std::shared_ptr<Dataset> Dataset::create(const std::map<std::string, std::vector<std::shared_ptr<Tensor>>> &inputs,
                                             const std::map<std::string, std::vector<std::shared_ptr<Tensor>>> &targets)
    {
        return std::make_shared<Dataset>(inputs, targets);
    }

    void Dataset::reset()
    {
        size_t dataset_size = input_tensors.begin()->second.size();

        indices.resize(dataset_size);
        for (size_t i = 0; i < dataset_size; i++)
        {
            indices[i] = i;
        }

        if (shuffle)
        {
            std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(indices.begin(), indices.end(), g);
        }

        current_pos = 0;
    }

    bool Dataset::has_next() const
    {
        return current_pos < indices.size();
    }

    std::pair<std::map<std::string, std::shared_ptr<Tensor>>,
              std::map<std::string, std::shared_ptr<Tensor>>>
    Dataset::next()
    {
        std::map<std::string, std::shared_ptr<Tensor>> batch_inputs;
        std::map<std::string, std::shared_ptr<Tensor>> batch_targets;

        // Just grab the tensors at the current index
        for (const auto &[key, tensor_vec] : input_tensors)
        {
            batch_inputs[key] = tensor_vec[indices[current_pos]];
        }

        for (const auto &[key, tensor_vec] : target_tensors)
        {
            batch_targets[key] = tensor_vec[indices[current_pos]];
        }

        current_pos++; // Move to next position
        return {batch_inputs, batch_targets};
    }
}