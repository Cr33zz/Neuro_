#pragma once

#include <driver_types.h>
#include <string>

#include "Types.h"

namespace Neuro
{
    using namespace std;

    enum EStorageType
    {
        ST_Default = 0,
        ST_RefCounted = 1 << 1,
        ST_Offloadable = 1 << 2,
        ST_KeepDevMem = 1 << 3,
    };

    class Storage
    {
    public:
        Storage(int type = ST_Default, size_t size = 0, const string& name = "");
        Storage(const Storage& other);
        Storage& operator=(const Storage& other);
        ~Storage();

        void ChangeType(int type);
        void Resize(size_t size);

        void Allocate();
        void Free();

        void AllocateOnDevice() const;
        void FreeOnDevice();

        void Offload() const;
        void Prefetch() const;

        ELocation Location() const { return m_Location; }

        void CopyToDevice() const;
        void CopyToHost() const;
        void CopyD2D(void* destDevPtr) const;

        void OverrideHost();
        void OverrideDevice();

        void IncRef(size_t n);
        void DecRef(size_t n);

        const float* Data() const { return m_DataPtr; }
        const float* DataEnd() const { return m_DataPtr + m_Size; }
        const float* DeviceData() const { return m_DeviceDataPtr; }
        float* Data();
        float* DeviceData();

        size_t Size() const { return m_Size; }
        size_t SizeInBytes() const { return m_Size * sizeof(float); }
        size_t AllocSizeInBytes() const { return m_AllocSize * sizeof(float); }

    private:
        float* m_DataPtr = nullptr;
        float* m_DeviceDataPtr = nullptr;
        int m_Type = ST_Default;
        size_t m_AllocSize = 0;
        size_t m_Size = 0;
        int m_RefCount = 0;
        cudaEvent_t m_OffloadEvent = nullptr;
        cudaEvent_t m_PrefetchEvent = nullptr;
        mutable ELocation m_Location = None;
        string m_Name = "";
    };
}
