#include "CudaTensor.h"

namespace ryupy {
    namespace cuda {

        CudaTensor::CudaTensor(int size) : ITensor(size) {
            data = new float[size];
            for (int i = 0; i < size; ++i) {
                data[i] = 2.0f;
            }
        }

        CudaTensor* CudaTensor::operator+(const ITensor &other) const {
            const CudaTensor *cuda_other = dynamic_cast<const CudaTensor *>(&other);
            if (!cuda_other) {
                throw std::invalid_argument("Unexpected tensor type.");
            }

            CudaTensor *result = new CudaTensor(size);
            for (int i = 0; i < size; ++i) {
                result->data[i] = this->data[i] + cuda_other->data[i];
            }
            return result;
        }

        void CudaTensor::printInfo() const {
            std::cout << "CudaTensor with size: " << size << " and data: ";
            for (int i = 0; i < size; ++i) {
                std::cout << data[i] << " ";
            }
            std::cout << std::endl;
        }

    } // namespace cuda
} // namespace ryupy
