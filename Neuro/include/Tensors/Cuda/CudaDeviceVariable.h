#pragma once
#include "Types.h"

#ifdef CUDA_ENABLED
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#endif

namespace Neuro
{
    using namespace std;

    template<typename T> 
    class CudaDeviceVariable
    {
    public:
        CudaDeviceVariable(size_t length)
        {
#ifdef CUDA_ENABLED
            m_Length = length;
            m_TypeSize = sizeof(T);
            m_IsOwner = true;
            cudaMalloc(&m_DevPtr, GetSizeInBytes());
#endif
        }

        CudaDeviceVariable(const CudaDeviceVariable<T>& var, size_t lengthOffset)
        {
#ifdef CUDA_ENABLED
            m_DevPtr = (T*)var.m_DevPtr + lengthOffset;
            m_TypeSize = var.m_TypeSize;
            m_Length = var.m_Length - lengthOffset;
            m_IsOwner = false;
#endif
        }

        CudaDeviceVariable(const CudaDeviceVariable<T>* var, size_t lengthOffset)
            : CudaDeviceVariable(*var, lengthOffset)
        {
        }

        ~CudaDeviceVariable()
        {
#ifdef CUDA_ENABLED
            if (m_IsOwner)
                cudaFree(m_DevPtr);
#endif
        }

        void CopyToDevice(const T* source) const
        {
#ifdef CUDA_ENABLED
            cudaMemcpy(m_DevPtr, source, GetSizeInBytes(), cudaMemcpyHostToDevice);
#endif
        }

        void CopyToDevice(const vector<T>& source) const
        {
            CopyToDevice(&source[0]);
        }

        void CopyToHost(T* dest) const
        {
#ifdef CUDA_ENABLED
            cudaMemcpy(dest, m_DevPtr, GetSizeInBytes(), cudaMemcpyDeviceToHost);
#endif
        }

        void CopyToHost(vector<T>& dest) const
        {
            CopyToHost(&dest[0]);
        }

        T* GetDevicePtr() const { return static_cast<T*>(m_DevPtr); }
        size_t GetSizeInBytes() const { return m_Length * m_TypeSize; }

    private:
        void* m_DevPtr;
        size_t m_Length = 0;
        size_t m_TypeSize = 0;
        bool m_IsOwner = false;
    };
}
