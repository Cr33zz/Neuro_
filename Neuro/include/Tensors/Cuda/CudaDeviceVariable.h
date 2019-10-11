#pragma once

#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

#include "Types.h"
#include "Tensors/Cuda/CudaErrorCheck.h"

namespace Neuro
{
    using namespace std;

    template<typename T> 
    class CudaDeviceVariable
    {
    public:
        CudaDeviceVariable(size_t length)
        {
            m_AllocatedLength = m_Length = length;
            m_TypeSize = sizeof(T);
            m_IsOwner = true;
            //CUDA_CHECK(cudaMalloc(&m_DevPtr, GetAllocatedSizeInBytes()));
            cudaMalloc(&m_DevPtr, GetAllocatedSizeInBytes());
            if (m_AllocatedLength > 0 && !m_DevPtr)
            {
                m_IsHostAlloc = true;
                CUDA_CHECK(cudaHostAlloc(&m_DevPtr, GetAllocatedSizeInBytes(), cudaHostAllocMapped));
            }
            NEURO_ASSERT(m_AllocatedLength == 0 || m_DevPtr, "Failed to allocate GPU memory.");
        }

        CudaDeviceVariable(const CudaDeviceVariable<T>& var, size_t lengthOffset)
        {
            m_DevPtr = (T*)var.m_DevPtr + lengthOffset;
            m_TypeSize = var.m_TypeSize;
            assert(var.m_Length > lengthOffset);
            m_Length = var.m_Length - lengthOffset;
            m_IsOwner = false;
        }

        CudaDeviceVariable(const CudaDeviceVariable<T>* var, size_t lengthOffset)
            : CudaDeviceVariable(*var, lengthOffset)
        {
        }

        ~CudaDeviceVariable()
        {
            if (m_IsOwner)
            {
                if (m_IsHostAlloc)
                    CUDA_CHECK(cudaFreeHost(m_DevPtr));
                else
                    CUDA_CHECK(cudaFree(m_DevPtr));
            }
        }

        bool Resize(size_t length)
        {
            assert(m_IsOwner);

            if (length <= m_AllocatedLength)
            {
                m_Length = length;
                return false;            
            }
            
            m_Length = m_AllocatedLength = length;
            CUDA_CHECK(cudaFree(m_DevPtr));
            CUDA_CHECK(cudaMalloc(&m_DevPtr, GetAllocatedSizeInBytes()));
            NEURO_ASSERT(m_AllocatedLength == 0 || m_DevPtr, "Failed to allocate GPU memory.");
            return true;
        }

        void CopyToDevice(const T* source) const
        {
            CUDA_CHECK(cudaMemcpy(m_DevPtr, source, GetSizeInBytes(), cudaMemcpyHostToDevice));
        }

        void CopyToDevice(const vector<T>& source) const
        {
            CopyToDevice(&source[0]);
        }

        void CopyToHost(T* dest) const
        {
            CUDA_CHECK(cudaMemcpy(dest, m_DevPtr, GetSizeInBytes(), cudaMemcpyDeviceToHost));
        }

        void CopyToHost(vector<T>& dest) const
        {
            CopyToHost(&dest[0]);
        }

        void CopyTo(void* destDevPtr) const
        {
            CUDA_CHECK(cudaMemcpy(destDevPtr, m_DevPtr, GetSizeInBytes(), cudaMemcpyDeviceToDevice));
        }

        void ZeroOnDevice() const
        {
            CUDA_CHECK(cudaMemset(m_DevPtr, 0, GetSizeInBytes()));
        }

        void OneOnDevice() const
        {
            //CUDA_CHECK(cuMemsetD32(m_DevPtr, 1.f, GetSizeInBytes()));
        }

        T* GetDevicePtr() const { return static_cast<T*>(m_DevPtr); }
        size_t GetSizeInBytes() const { return m_Length * m_TypeSize; }
        size_t GetAllocatedSizeInBytes() const { return m_AllocatedLength * m_TypeSize; }

    private:
        void* m_DevPtr = nullptr;
        size_t m_Length = 0;
        size_t m_AllocatedLength = 0;
        size_t m_TypeSize = 0;
        bool m_IsOwner = false;
        bool m_IsHostAlloc = false;
    };
}
