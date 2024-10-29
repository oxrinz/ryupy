#include "CpuTensor.h"

namespace ryupy
{
    namespace cpu
    {
        CpuTensor::CpuTensor(int size) : ITensor(size)
        {
            data = new float[size];
            for (int i = 0; i < size; ++i)
            {
                data[i] = 1.0f;
            }
        }

        CpuTensor *CpuTensor::operator+(const ITensor &other) const
        {
            const CpuTensor *cpu_other = dynamic_cast<const CpuTensor *>(&other);
            if (!cpu_other)
            {
                throw std::invalid_argument("Unexpected tensor type.");
            }

            CpuTensor *result = new CpuTensor(size);
            for (int i = 0; i < size; ++i)
            {
                result->data[i] = this->data[i] + cpu_other->data[i];
            }
            return result;
        }

        void CpuTensor::printInfo() const
        {
            std::cout << "CpuTensor with size: " << size << " and data: ";
            for (int i = 0; i < size; ++i)
            {
                std::cout << data[i] << " ";
            }
            std::cout << std::endl;
        }
    }
}