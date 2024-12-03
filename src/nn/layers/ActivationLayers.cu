#include "ActivationLayers.h"
#include <unordered_map>
#include <iostream>

namespace ryupy
{
    namespace nn
    {
        namespace detail
        {
            cudnnHandle_t &get_cudnn_handle()
            {
                static cudnnHandle_t handle = nullptr;
                if (handle == nullptr)
                {
                    cudnnCreate(&handle);
                }
                return handle;
            }

            struct pair_hash
            {
                template <class T1, class T2>
                std::size_t operator()(const std::pair<T1, T2> &p) const
                {
                    auto h1 = std::hash<T1>{}(p.first);
                    auto h2 = std::hash<T2>{}(p.second);
                    return h1 ^ (h2 << 1);
                }
            };

            struct ActivationDescriptor
            {
                cudnnActivationDescriptor_t desc;

                ActivationDescriptor()
                {
                    cudnnCreateActivationDescriptor(&desc);
                }

                ~ActivationDescriptor()
                {
                    cudnnDestroyActivationDescriptor(desc);
                }

                ActivationDescriptor(const ActivationDescriptor &) = delete;
                ActivationDescriptor &operator=(const ActivationDescriptor &) = delete;
            };

            cudnnActivationDescriptor_t get_activation_descriptor(
                cudnnActivationMode_t mode,
                float param = 0.0f)
            {
                static std::unordered_map<std::pair<cudnnActivationMode_t, float>,
                                          ActivationDescriptor,
                                          pair_hash>
                    descriptors;

                auto key = std::make_pair(mode, param);
                auto it = descriptors.find(key);

                if (it == descriptors.end())
                {
                    auto &descriptor = descriptors[key];
                    cudnnSetActivationDescriptor(descriptor.desc,
                                                 mode,
                                                 CUDNN_PROPAGATE_NAN,
                                                 param);
                    return descriptor.desc;
                }
                return it->second.desc;
            }

            std::shared_ptr<Tensor> activation_forward(
                Tensor &input,
                cudnnActivationDescriptor_t activation_desc)
            {
                auto output = std::make_shared<Tensor>(input.shape);
                float alpha = 1.0f, beta = 0.0f;

                cudnnActivationForward(get_cudnn_handle(),
                                       activation_desc,
                                       &alpha,
                                       input.tensor_desc,
                                       input.d_data,
                                       &beta,
                                       output->tensor_desc,
                                       output->d_data);

                if (input.requires_grad)
                {
                    output->requires_grad = true;
                    output->is_leaf = false;
                    output->prev = {std::make_shared<Tensor>(input)};
                    output->backward_fn = [output, &input, activation_desc]()
                    {
                        if (!input.grad)
                        {
                            input.grad = std::make_shared<Tensor>(input.shape);
                        }

                        input.grad = output->grad->copy();

                        if (input.requires_grad)
                        {
                            float alpha = 1.0f, beta = 0.0f;
                            cudnnActivationBackward(get_cudnn_handle(),
                                                    activation_desc,
                                                    &alpha,
                                                    output->tensor_desc,
                                                    output->d_data,
                                                    output->tensor_desc,
                                                    output->grad->d_data,
                                                    input.tensor_desc,
                                                    input.d_data,
                                                    &beta,
                                                    input.tensor_desc,
                                                    input.grad->d_data);
                        }
                    };
                }
                return output;
            }
        }

        std::shared_ptr<Tensor> relu(Tensor &input)
        {
            auto activation_desc = detail::get_activation_descriptor(CUDNN_ACTIVATION_RELU);
            return detail::activation_forward(input, activation_desc);
        }

        std::shared_ptr<Tensor> sigmoid(Tensor &input)
        {
            auto activation_desc = detail::get_activation_descriptor(CUDNN_ACTIVATION_SIGMOID);
            return detail::activation_forward(input, activation_desc);
        }

        std::shared_ptr<Tensor> tanh(Tensor &input)
        {
            auto activation_desc = detail::get_activation_descriptor(CUDNN_ACTIVATION_TANH);
            return detail::activation_forward(input, activation_desc);
        }

        std::shared_ptr<Tensor> leaky_relu(Tensor &input, float negative_slope)
        {
            auto activation_desc = detail::get_activation_descriptor(CUDNN_ACTIVATION_CLIPPED_RELU, negative_slope);
            return detail::activation_forward(input, activation_desc);
        }

    } // namespace nn
} // namespace ryupy