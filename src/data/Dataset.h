#pragma once
#include "../tensors/Tensor.h"
namespace ryupy
{
    class Dataset
    {
    public:
        Dataset() = default;
        Dataset(const std::map<std::string, std::vector<std::shared_ptr<Tensor>>> &inputs,
                const std::map<std::string, std::vector<std::shared_ptr<Tensor>>> &targets);

        static std::shared_ptr<Dataset> create(const std::map<std::string, std::vector<std::shared_ptr<Tensor>>> &inputs,
                                               const std::map<std::string, std::vector<std::shared_ptr<Tensor>>> &targets);

        // Iterator methods
        void reset();          // Reset iteration
        bool has_next() const; // Check if more batches exist
        std::pair<std::map<std::string, std::shared_ptr<Tensor>>,
                  std::map<std::string, std::shared_ptr<Tensor>>>
        next(); // Get next batch

        std::map<std::string, std::vector<std::shared_ptr<Tensor>>> input_tensors;
        std::map<std::string, std::vector<std::shared_ptr<Tensor>>> target_tensors;
        int batch_size = 32;
        bool shuffle = true;
        std::vector<size_t> indices;
        size_t current_pos = 0;
    };
}