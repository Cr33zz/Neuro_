#pragma once

#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

#include "Types.h"
#include "Memory/MemoryManager.h"
#include "Tensors/Cuda/CudaErrorCheck.h"

#if 1
#define ALLOC MemoryManager::Default().Allocate
#define FREE 
#else
#define ALLOC cudaMalloc
#define FREE cudaFree
#endif

namespace Neuro
{
    using namespace std;

    template<typename T> 
    class CudaDeviceVariable
    {
    public:
        CudaDeviceVariable(size_t length, EOffloadMode offloadMode)
        {
            m_OffloadMode = offloadMode;
            m_AllocatedLength = m_Length = length;
            m_TypeSize = sizeof(T);
            m_IsOwner = true;
            Allocate();
            if (m_OffloadMode == Offload_Enabled)
            {
                CUDA_CHECK(MemoryManager::Default().AllocateForOffload(&m_HostPtr, GetAllocatedSizeInBytes()));
                CUDA_CHECK(cudaEventCreate(&m_OffloadEvent));
                CUDA_CHECK(cudaEventCreate(&m_PrefetchEvent));
            }
        }

        CudaDeviceVariable(const CudaDeviceVariable<T>& var, size_t lengthOffset)
        {
            m_DevPtr = (T*)var.m_DevPtr + lengthOffset;
            m_TypeSize = var.m_TypeSize;
            NEURO_ASSERT(var.m_Length > lengthOffset, "");
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
                Release();

                if (m_HostPtr)
                {
                    CUDA_CHECK(MemoryManager::Default().ReleaseForOffload(m_HostPtr));
                    CUDA_CHECK(cudaEventDestroy(m_OffloadEvent));
                    CUDA_CHECK(cudaEventDestroy(m_PrefetchEvent));
                }
            }
        }

        void Allocate() const
        {
            if (m_DevPtr)
                return;

            CUDA_CHECK(MemoryManager::Default().Allocate(&m_DevPtr, GetAllocatedSizeInBytes()));            
        }

        void Release()
        {
            NEURO_ASSERT(m_IsOwner, "Trying to release CUDA variable when not being owner.");
            if (!m_DevPtr)
                return;

            MemoryManager::Default().WaitForMemEvent(m_OffloadEvent);
            CUDA_CHECK(MemoryManager::Default().Release(m_DevPtr));
            m_DevPtr = nullptr;
        }

        bool Resize(size_t length)
        {
            NEURO_ASSERT(m_IsOwner, "Trying to resize CUDA variable when not being owner.");

            if (length <= m_AllocatedLength)
            {
                m_Length = length;
                return false;            
            }
            
            m_Length = m_AllocatedLength = length;
            CUDA_CHECK(MemoryManager::Default().Release(m_DevPtr));
            CUDA_CHECK(MemoryManager::Default().Allocate(&m_DevPtr, GetAllocatedSizeInBytes()));
            if (m_OffloadMode == Offload_Enabled)
            {
                CUDA_CHECK(MemoryManager::Default().ReleaseForOffload(m_HostPtr));
                CUDA_CHECK(MemoryManager::Default().AllocateForOffload(&m_HostPtr, GetAllocatedSizeInBytes()));
            }
            NEURO_ASSERT(m_AllocatedLength == 0 || m_DevPtr, "Failed to allocate GPU memory.");
            return true;
        }

        bool IsOffloaded()
        {
            return m_OffloadMode == Offload_Enabled && !m_DevPtr && cudaEventQuery(m_OffloadEvent) == cudaSuccess;
        }

        void Offload()
        {
            if (m_OffloadMode == Offload_Enabled)
                CUDA_CHECK(MemoryManager::Default().Offload(m_HostPtr, m_DevPtr, GetSizeInBytes(), m_OffloadEvent));
        }

        void Prefetch()
        {
            if (m_OffloadMode == Offload_Enabled)
            {
                Allocate();
                CUDA_CHECK(MemoryManager::Default().Prefetch(m_DevPtr, m_HostPtr, GetSizeInBytes(), m_PrefetchEvent));
            }
        }

        void CopyToDevice(const T* source) const
        {
            Allocate();

            if (m_OffloadMode == Offload_Enabled)
            {
                MemoryManager::Default().WaitForMemEvent(m_OffloadEvent);
                MemoryManager::Default().WaitForMemEvent(m_PrefetchEvent);
            }
            else
                CUDA_CHECK(cudaMemcpy(m_DevPtr, source, GetSizeInBytes(), cudaMemcpyHostToDevice));
        }

        void CopyToDevice(const vector<T>& source) const
        {
            CopyToDevice(&source[0]);
        }

        void CopyToHost(T* dest) const
        {
            if (m_OffloadMode == Offload_Enabled)
            {
                MemoryManager::Default().WaitForMemEvent(m_PrefetchEvent);
                CUDA_CHECK(cudaMemcpy(dest, m_HostPtr, GetSizeInBytes(), cudaMemcpyDeviceToHost));
            }
            else
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
        mutable void* m_DevPtr = nullptr;
        void* m_HostPtr = nullptr;
        cudaEvent_t m_OffloadEvent = nullptr;
        cudaEvent_t m_PrefetchEvent = nullptr;
        size_t m_Length = 0;
        size_t m_AllocatedLength = 0;
        size_t m_TypeSize = 0;
        bool m_IsOwner = false;
        EOffloadMode m_OffloadMode;
    };
}
