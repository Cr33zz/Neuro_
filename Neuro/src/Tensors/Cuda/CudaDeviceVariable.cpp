#include <cuda.h>
#include <cuda_runtime.h>

#include "Tensors/Cuda/CudaDeviceVariable.h"
#include "Memory/MemoryManager.h"
#include "Tensors/Cuda/CudaErrorCheck.h"

//#define ENABLE_CUDA_VAR_LOGS

#ifdef ENABLE_CUDA_VAR_LOGS
#include <windows.h>
#include <debugapi.h>
#define CUDA_VAR_DEBUG_INFO(...) do { static char buffer[1024]; sprintf(buffer, __VA_ARGS__); OutputDebugString(buffer); } while(0)
#else
#define CUDA_VAR_DEBUG_INFO(...) void
#endif

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    template<typename T>
    Neuro::CudaDeviceVariable<T>::CudaDeviceVariable(size_t length, EOffloadMode offloadMode, const string& name)
        : m_Name(name)
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

    //////////////////////////////////////////////////////////////////////////
    template<typename T>
    Neuro::CudaDeviceVariable<T>::CudaDeviceVariable(const CudaDeviceVariable<T>& var, size_t lengthOffset)
    {
        m_DevPtr = (T*)var.m_DevPtr + lengthOffset;
        m_TypeSize = var.m_TypeSize;
        NEURO_ASSERT(var.m_Length > lengthOffset, "");
        m_Length = var.m_Length - lengthOffset;
        m_IsOwner = false;
    }

    //////////////////////////////////////////////////////////////////////////
    template<typename T>
    Neuro::CudaDeviceVariable<T>::CudaDeviceVariable(const CudaDeviceVariable<T>* var, size_t lengthOffset)
        : CudaDeviceVariable(*var, lengthOffset)
    {
    }

    //////////////////////////////////////////////////////////////////////////
    template<typename T>
    Neuro::CudaDeviceVariable<T>::~CudaDeviceVariable()
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

    //////////////////////////////////////////////////////////////////////////
    template<typename T>
    void Neuro::CudaDeviceVariable<T>::Allocate() const
    {
        CUDA_VAR_DEBUG_INFO("Allocating '%s' ", m_Name.c_str());
        if (m_DevPtr)
        {
            CUDA_VAR_DEBUG_INFO("<<< already allocated.\n");
            return;
        }
        CUDA_VAR_DEBUG_INFO("<<< allocating.\n");
        CUDA_CHECK(MemoryManager::Default().Allocate(&m_DevPtr, GetAllocatedSizeInBytes(), m_Name.c_str()));
    }

    //////////////////////////////////////////////////////////////////////////
    template<typename T>
    void Neuro::CudaDeviceVariable<T>::Release()
    {
        CUDA_VAR_DEBUG_INFO("Releasing '%s' ", m_Name.c_str());
        NEURO_ASSERT(m_IsOwner, "Trying to release CUDA variable when not being owner.");
        if (!m_DevPtr)
        {
            CUDA_VAR_DEBUG_INFO("<<< already released.\n");
            return;
        }

        CUDA_VAR_DEBUG_INFO("<<< release incoming.\n");
        MemoryManager::Default().WaitForMemEvent(m_OffloadEvent);
        CUDA_CHECK(MemoryManager::Default().Release(m_DevPtr));
        m_DevPtr = nullptr;
    }

    //////////////////////////////////////////////////////////////////////////
    template<typename T>
    bool Neuro::CudaDeviceVariable<T>::Resize(size_t length)
    {
        NEURO_ASSERT(m_IsOwner, "Trying to resize CUDA variable when not being owner.");

        if (length <= m_AllocatedLength)
        {
            m_Length = length;
            return false;
        }

        m_Length = m_AllocatedLength = length;
        CUDA_CHECK(MemoryManager::Default().Release(m_DevPtr));
        CUDA_CHECK(MemoryManager::Default().Allocate(&m_DevPtr, GetAllocatedSizeInBytes(), m_Name.c_str()));
        if (m_OffloadMode == Offload_Enabled)
        {
            CUDA_CHECK(MemoryManager::Default().ReleaseForOffload(m_HostPtr));
            CUDA_CHECK(MemoryManager::Default().AllocateForOffload(&m_HostPtr, GetAllocatedSizeInBytes()));
        }
        NEURO_ASSERT(m_AllocatedLength == 0 || m_DevPtr, "Failed to allocate GPU memory.");
        return true;
    }

    //////////////////////////////////////////////////////////////////////////
    template<typename T>
    bool Neuro::CudaDeviceVariable<T>::IsOffloaded()
    {
        return m_OffloadMode == Offload_Enabled && !m_DevPtr && cudaEventQuery(m_OffloadEvent) == cudaSuccess;
    }

    //////////////////////////////////////////////////////////////////////////
    template<typename T>
    void Neuro::CudaDeviceVariable<T>::Offload()
    {
        if (!m_AllocatedLength)
            return;

        CUDA_VAR_DEBUG_INFO("Offloading '%s'[%s] ", m_Name.c_str(), ToString(m_OffloadMode));
        if (m_OffloadMode == Offload_Enabled)
        {
            if (!m_DevPtr)
            {
                CUDA_VAR_DEBUG_INFO("<<< nothing to offload.\n");
                return;
            }
            CUDA_VAR_DEBUG_INFO("<<< requested.\n");
            CUDA_CHECK(MemoryManager::Default().Offload(m_HostPtr, m_DevPtr, GetSizeInBytes(), m_OffloadEvent));
        }
        else
            CUDA_VAR_DEBUG_INFO("<<< not supported.\n");
    }

    //////////////////////////////////////////////////////////////////////////
    template<typename T>
    void Neuro::CudaDeviceVariable<T>::Prefetch()
    {
        if (!m_AllocatedLength)
            return;

        if (m_OffloadMode == Offload_Enabled)
        {
            NEURO_ASSERT(m_HostPtr, "Trying to prefetch unallocated device variable.");
            if (!m_HostPtr)
            {
                CUDA_VAR_DEBUG_INFO("Prefetching '%s'[%s] <<< nothing to preload.\n", m_Name.c_str(), ToString(m_OffloadMode));
                return;
            }

            if (!m_DevPtr)
                Allocate();

            if (cudaEventQuery(m_PrefetchEvent) != cudaSuccess)
            {
                CUDA_VAR_DEBUG_INFO("Prefetching '%s'[%s] <<< requested already.\n", m_Name.c_str(), ToString(m_OffloadMode));
            }
            else
            {
                CUDA_VAR_DEBUG_INFO("Prefetching '%s'[%s] <<< requested.\n", m_Name.c_str(), ToString(m_OffloadMode));
                CUDA_CHECK(MemoryManager::Default().Prefetch(m_DevPtr, m_HostPtr, GetSizeInBytes(), m_PrefetchEvent));
            }
        }
        else
            CUDA_VAR_DEBUG_INFO("Prefetching '%s'[%s] <<< not supported.\n", m_Name.c_str(), ToString(m_OffloadMode));
    }

    //////////////////////////////////////////////////////////////////////////
    template<typename T>
    void Neuro::CudaDeviceVariable<T>::CopyToDevice(const T* source, ELocation currLocation) const
    {
        bool wasDevMemoryAllocated = true;

        if (!m_DevPtr)
        {
            Allocate();
            wasDevMemoryAllocated = false;
        }

        if (currLocation == Host)
        {
            CUDA_VAR_DEBUG_INFO("Copy to device '%s'[%s]\n", m_Name.c_str(), ToString(m_OffloadMode));
            CUDA_CHECK(cudaMemcpy(m_DevPtr, source, GetSizeInBytes(), cudaMemcpyHostToDevice));
        }
        else if (m_OffloadMode == Offload_Enabled)
        {
            if (!wasDevMemoryAllocated)
            {
                CUDA_VAR_DEBUG_INFO("Copy to device '%s'[%s] <<< allocated device memory - copy from host ptr\n", m_Name.c_str(), ToString(m_OffloadMode));
                CUDA_CHECK(cudaMemcpy(m_DevPtr, m_HostPtr, GetSizeInBytes(), cudaMemcpyHostToDevice));
            }
            else
            {
                CUDA_VAR_DEBUG_INFO("Copy to device '%s'[%s] <<< prefetch completed check only\n", m_Name.c_str(), ToString(m_OffloadMode));
                // we don't have to wait for offload because a valid copy still exists in GPU memory
                /*if (cudaEventQuery(m_OffloadEvent) == cudaErrorNotReady)
                    CUDA_VAR_DEBUG_INFO("Waiting for offload... '%s'[%s]\n", m_Name.c_str(), ToString(m_OffloadMode));
                MemoryManager::Default().WaitForMemEvent(m_OffloadEvent);*/
                // we have to make sure copy to device is blocked until prefetch is done
                if (cudaEventQuery(m_PrefetchEvent) == cudaErrorNotReady)
                    CUDA_VAR_DEBUG_INFO("Waiting for prefetch... '%s'[%s]\n", m_Name.c_str(), ToString(m_OffloadMode));
                MemoryManager::Default().WaitForMemEvent(m_PrefetchEvent);
            }
        }
    }

    //////////////////////////////////////////////////////////////////////////
    template<typename T>
    void Neuro::CudaDeviceVariable<T>::CopyToDevice(const vector<T>& source, ELocation currLocation) const
    {
        CopyToDevice(&source[0], currLocation);
    }

    //////////////////////////////////////////////////////////////////////////
    template<typename T>
    void Neuro::CudaDeviceVariable<T>::CopyToHost(T* dest) const
    {
        CUDA_VAR_DEBUG_INFO("Copy to host '%s'[%s]\n", m_Name.c_str(), ToString(m_OffloadMode));
        if (m_OffloadMode == Offload_Enabled)
        {
            if (cudaEventQuery(m_OffloadEvent) == cudaErrorNotReady)
                CUDA_VAR_DEBUG_INFO("Waiting for offload... '%s'[%s]\n", m_Name.c_str(), ToString(m_OffloadMode));
            MemoryManager::Default().WaitForMemEvent(m_OffloadEvent);
            CUDA_CHECK(cudaMemcpy(dest, m_HostPtr, GetSizeInBytes(), cudaMemcpyDeviceToHost));
        }
        else
            CUDA_CHECK(cudaMemcpy(dest, m_DevPtr, GetSizeInBytes(), cudaMemcpyDeviceToHost));
    }

    //////////////////////////////////////////////////////////////////////////
    template<typename T>
    void Neuro::CudaDeviceVariable<T>::CopyToHost(vector<T>& dest) const
    {
        CopyToHost(&dest[0]);
    }

    //////////////////////////////////////////////////////////////////////////
    template<typename T>
    void Neuro::CudaDeviceVariable<T>::OverrideHost() const
    {
        CUDA_VAR_DEBUG_INFO("Override host '%s'[%s]\n", m_Name.c_str(), ToString(m_OffloadMode));
    }

    //////////////////////////////////////////////////////////////////////////
    template<typename T>
    void Neuro::CudaDeviceVariable<T>::OverrideDevice() const
    {
        CUDA_VAR_DEBUG_INFO("Override device '%s'[%s]\n", m_Name.c_str(), ToString(m_OffloadMode));
    }

    //////////////////////////////////////////////////////////////////////////
    template<typename T>
    void Neuro::CudaDeviceVariable<T>::CopyTo(void* destDevPtr) const
    {
        NEURO_ASSERT(destDevPtr, "Invalid destination pointer.");
        CUDA_CHECK(cudaMemcpy(destDevPtr, m_DevPtr, GetSizeInBytes(), cudaMemcpyDeviceToDevice));
    }

    template CudaDeviceVariable<float>;
    template CudaDeviceVariable<char>;
}