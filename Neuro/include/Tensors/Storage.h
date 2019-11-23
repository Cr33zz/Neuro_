#pragma once

#include <driver_types.h>
#include <future>
#include <mutex>
#include <string>

#include "Types.h"

namespace Neuro
{
    using namespace std;

    enum EStorageType
    {
        ST_Default = 0,
        ST_RefCounted = 1 << 1,
        ST_DeviceRefCounted = 1 << 1,
        ST_Offloadable = 1 << 2,
        ST_KeepDevMem = 1 << 3,
    };

    class Storage
    {
    public:
        Storage(int type = ST_Default, size_t size = 0, const string& name = "");
        Storage(const Storage& other);
        Storage(Storage&& other);
        Storage& operator=(const Storage& other);
        Storage& operator=(Storage&& other);
        ~Storage();

        void ChangeType(int type);
        void Resize(size_t size);
        void Rename(const string& name);
        /// Deallocates all memory on both host and device. Location will be changed to None. Size will remain unchanged.
        void Release();

        void AllocateOnHost() const;
        void FreeOnHost();

        void AllocateOnDevice() const;
        void FreeOnDevice(bool force = false, bool forceWaitForOffload = false);

        void Offload(bool force) const;
        void ScheduleOffload() const;
        void Preload() const;

        ELocation Location() const { return m_DataLocation; }

        void CopyToDevice() const;
        void CopyToHost(bool allowAlloc = false) const;
        void SyncToHost() const;
        void CopyWithinDevice(void* destDevPtr) const;
        void CopyWithinDevice(void* destDevPtr, const void* srcDevPtr, size_t sizeInBytes) const;

        void OverrideHost();
        void OverrideDevice();

        void ResetDeviceRef(size_t n);
        void IncDeviceRef(size_t n) const;
        void DecDeviceRef(size_t n);

        void ResetRef(size_t n);
        void IncRef(size_t n) const;
        void DecRef(size_t n);

        const float* Data() const;
        const float* DataUnsafe() const { return m_DataPtr; }
        const float* DataEnd() const { return m_DataPtr + m_Size; }
        const float* DeviceData() const;
        float* Data();
        float* DeviceData();

        bool IsHostAllocated() const { return m_DataPtr != nullptr; }
        bool IsDeviceAllocated() const { return m_DeviceDataPtr != nullptr; }

        size_t Size() const { return m_Size; }
        size_t SizeInBytes() const { return m_Size * sizeof(float); }
        size_t AllocSizeInBytes() const { return m_AllocSize * sizeof(float); }

    private:
        static void OffloadTriggerCallback(void* userData);
        static void OffloadDoneCallback(void* userData);
        static void PreloadDoneCallback(void* userData);

        void WaitForOffload() const;
        void WaitForPreload() const;

        float* m_DataPtr = nullptr;
        float* m_DeviceDataPtr = nullptr;
        int m_Type = ST_Default;
        size_t m_AllocSize = 0;
        size_t m_Size = 0;
        mutable int m_DeviceDataRefCount = 0;
        mutable int m_DataRefCount = 0;
        cudaEvent_t m_OffloadEvent = nullptr;
        mutable bool m_OffloadDone = false;
        mutable mutex m_OffloadDoneCallbackMtx;
        mutable bool m_FreeDeviceMemOnOffloadDone = false;
        mutable bool m_FreePinnedMemOnOffloadDone = false;
        mutable bool m_OffloadRequested = false;
        mutable promise<void> m_OffloadPromise;
        mutable future<void> m_OffloadFuture;
        mutable promise<void> m_PreloadPromise;
        mutable future<void> m_PreloadFuture;
        mutable bool m_PreloadRequested = false;
        cudaEvent_t m_PreloadEvent = nullptr;
        mutable ELocation m_DataLocation = None;
        string m_Name = "";
    };
}
