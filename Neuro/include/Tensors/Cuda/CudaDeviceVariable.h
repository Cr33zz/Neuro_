#pragma once

#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

namespace Neuro
{
    using namespace std;

    template<typename T> 
    class CudaDeviceVariable
    {
    public:
        CudaDeviceVariable(size_t length)
        {
            m_Length = length;
            m_TypeSize = sizeof(T);
            m_IsOwner = true;
            cudaMalloc(&m_DevPtr, GetSizeInBytes());
        }

        CudaDeviceVariable(const CudaDeviceVariable<T>& var, size_t lengthOffset)
        {
            m_DevPtr = (T*)var.m_DevPtr + lengthOffset * var.m_TypeSize;
            m_TypeSize = var.m_TypeSize;
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
                cudaFree(m_DevPtr);
        }

        void CopyToDevice(const T* source)
        {
            cudaMemcpy(m_DevPtr, source, GetSizeInBytes(), cudaMemcpyHostToDevice);
        }

        void CopyToDevice(const vector<T>& source)
        {
            CopyToDevice(&source[0]);
        }

        void CopyToHost(T* dest)
        {
            cudaMemcpy(dest, m_DevPtr, GetSizeInBytes(), cudaMemcpyDeviceToHost);
        }

        void CopyToHost(vector<T>& dest)
        {
            CopyToHost(&dest[0]);
        }

        void* GetDevicePtr() const { return m_DevPtr; }
        size_t GetSizeInBytes() const { return m_Length * m_TypeSize; }

        operator T*() const { return static_cast<T*>(m_DevPtr); }
        
    private:
        void* m_DevPtr;
        size_t m_Length = 0;
        size_t m_TypeSize = 0;
        bool m_IsOwner = false;
    };
}
