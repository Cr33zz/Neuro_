#pragma once

#include <vector>
#include <driver_types.h>

#include "Types.h"

namespace Neuro
{
    using namespace std;

    template<typename T> 
    class CudaDeviceVariable
    {
    public:
        CudaDeviceVariable(size_t length, EOffloadMode offloadMode, const string& name);
        CudaDeviceVariable(const CudaDeviceVariable<T>& var, size_t lengthOffset);
        CudaDeviceVariable(const CudaDeviceVariable<T>* var, size_t lengthOffset);
        ~CudaDeviceVariable();

        void Allocate() const;
        void Release();

        bool Resize(size_t length);

        bool IsOffloaded();
        void Offload();
        void Prefetch();

        void CopyToDevice(const T* source, ELocation currLocation) const;
        void CopyToDevice(const vector<T>& source, ELocation currLocation) const;
        void CopyToHost(T* dest) const;
        void CopyToHost(vector<T>& dest) const;

        void OverrideHost() const;
        void OverrideDevice() const;

        void CopyTo(void* destDevPtr) const;

        T* GetDevicePtr() const { return m_DevPtr ? static_cast<T*>(m_DevPtr) : static_cast<T*>(m_HostPtr); }
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
        string m_Name;
    };
}
