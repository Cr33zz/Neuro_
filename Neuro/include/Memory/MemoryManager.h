#pragma once

#include <cstdio>
#include <list>
#include <mutex>
#include <driver_types.h>

namespace Neuro
{
    using namespace std;

    enum EMemStatus
    {
        MEM_STATUS_SUCCESS = 0,
        MEM_STATUS_CUDA_ERROR,
        MEM_STATUS_INVALID_ARGUMENT,
        MEM_STATUS_NOT_INITIALIZED,
        MEM_STATUS_OUT_OF_MEMORY,
        MEM_STATUS_UNKNOWN_ERROR
    };

    enum EMemManagerFlag
    {
        MEM_FLAGS_DEFAULT = 0,       /// Default flags.
        MEM_FLAGS_CANNOT_GROW = 1,   /// Prevent the manager from growing its memory consumption.
    };

    // A node in the linked list of memory blocks
    class Block
    {
    public:
        Block(char* data, size_t size, Block *next, bool isHead) : m_Data(data) , m_Size(size) , m_Next(next) , m_IsHead(isHead) {}

        inline const char* GetData() const { return m_Data; }
        inline char* GetData() { return m_Data; }

        inline void SetSize(size_t size) { m_Size = size; }
        inline size_t GetSize() const { return m_Size; }

        inline void SetNext(Block *next) { m_Next = next; }
        inline const Block* GetNext() const { return m_Next; }
        inline Block* GetNext() { return m_Next; }

        inline void SetHeadFlag(bool isHead) { m_IsHead = isHead; }
        inline bool IsHead() const { return m_IsHead; }

        /// Debug annotation
        string m_Annotation;

    private:
        /// The pointer to the memory region on the device. 
        char* m_Data;
        /// The size of the memory buffer.
        size_t m_Size;
        /// The prev/next blocks in the linked list of blocks.
        Block* m_Next;        
        /// Is it a head node (i.e. a node obtained from system allocator.
        bool m_IsHead;
    };

    // Block of memory allocated by system call
    struct NativeBlock
    {
        void* ptr;
        size_t size;
    };

    class MemoryManagerBase
    {
    public:
        MemoryManagerBase(size_t allocGranularity, size_t nativeAllocGranularity);

        EMemStatus Allocate(void** ptr, size_t size, const string& annotation = "");
        EMemStatus ScheduleFree(void* ptr);
        EMemStatus Free(void* ptr);

        EMemStatus DumpMemoryState(const string& filename) const;
        EMemStatus DumpMemoryState(FILE* file) const;
        void UpdateAnnotation(void* ptr, const string& annotation);

        EMemStatus ReleaseAll();

    protected:
        virtual void InternalAllocate(void** ptr, size_t size, const string& annotation = "") = 0;
        virtual void InternalFree(void* ptr) = 0;
        virtual void InternalMemset(void* ptr, uint8_t value, size_t size) = 0;
        virtual const char* InternalName() const = 0;

        EMemStatus AllocateBlock(Block*& curr, Block*& prev, size_t size);

    private:
        EMemStatus ReleaseBlock(Block* curr, Block* prev);
        EMemStatus SplitBlock(Block* curr, Block* prev, size_t size);
        EMemStatus FindBestBlock(Block*& best, Block*& prev, size_t size);

        EMemStatus AddNativeBlock(void* ptr, size_t size);
        EMemStatus RemoveNativeBlock(void* ptr);
        
        EMemStatus PrintList(FILE* file, const char* name, const Block* head) const;
        inline EMemStatus GetUsedMemory(size_t& usedMemory) const;
        inline EMemStatus GetFreeMemory(size_t& freeMemory) const;
        EMemStatus GetMemory(size_t& size, const Block* head) const;        

        Block* m_UsedBlocks = nullptr;
        Block* m_FreeBlocks = nullptr;
        list<NativeBlock> m_NativeBlocks;
        uint32_t m_Flags = MEM_FLAGS_DEFAULT;
        const size_t m_AllocGranularity;
        const size_t m_NativeAllocGranularity;
        size_t m_AllocatedMemSize = 0;
        size_t m_AllocatedMemPeakSize = 0;
        vector<void*> m_ScheduledDeallocations;
        int m_MinSizeForDirectAllocation = -1;
        vector<void*> m_DirectAlocations;

        mutex m_AllocFreeMtx;
        mutex m_ScheduledFreeMtx;
    };

    // Memory manager for GPU memory
    class DeviceMemoryManager : public MemoryManagerBase
    {
    public:
        DeviceMemoryManager();
        static DeviceMemoryManager& Default();

        EMemStatus Reserve(size_t size);
        EMemStatus Offload(void* dst, void* src, size_t size, cudaEvent_t memEvent, cudaHostFn_t callback = nullptr, void* userData = nullptr);
        EMemStatus Preload(void* dst, void* src, size_t size, cudaEvent_t memEvent, cudaHostFn_t callback = nullptr, void* userData = nullptr);
        EMemStatus WaitForMemEvent(cudaEvent_t memEvent);
        EMemStatus ForceStreamsSync();
        EMemStatus ForceOffloadStreamSync();
        EMemStatus ForcePreloadStreamSync();

    protected:
        virtual void InternalAllocate(void** ptr, size_t size, const string& annotation = "") override;
        virtual void InternalFree(void* ptr) override;
        virtual void InternalMemset(void* ptr, uint8_t value, size_t size) override;
        virtual const char* InternalName() const override { return "Device"; }

    private:
        cudaStream_t m_OffloadStream = nullptr;
        cudaStream_t m_PreloadStream = nullptr;
        bool m_IsStreamBlocking = false;
    };

    // Memory manager for generic CPU memory
    class HostMemoryManager : public MemoryManagerBase
    {
    public:
        HostMemoryManager();
        static HostMemoryManager& Default();

    protected:
        virtual void InternalAllocate(void** ptr, size_t size, const string& annotation = "") override;
        virtual void InternalFree(void* ptr) override;
        virtual void InternalMemset(void* ptr, uint8_t value, size_t size) override;
        virtual const char* InternalName() const override { return "Host"; }
    };

    // Memory manager for pinned (unpageable) CPU memory
    class HostPinnedMemoryManager : public MemoryManagerBase
    {
    public:
        HostPinnedMemoryManager();
        static HostPinnedMemoryManager& Default();

    protected:
        virtual void InternalAllocate(void** ptr, size_t size, const string& annotation = "") override;
        virtual void InternalFree(void* ptr) override;
        virtual void InternalMemset(void* ptr, uint8_t value, size_t size) override;
        virtual const char* InternalName() const override { return "Host pinned"; }
    };

    ///
    static void DumpMemoryManagers(const string& filename)
    {
        FILE* file;
        fopen_s(&file, filename.c_str(), "w");
        DeviceMemoryManager::Default().DumpMemoryState(file);
        HostMemoryManager::Default().DumpMemoryState(file);
        HostPinnedMemoryManager::Default().DumpMemoryState(file);
        fclose(file);
    }

    static void ReleaseAllMemory()
    {
        DeviceMemoryManager::Default().ReleaseAll();
        HostMemoryManager::Default().ReleaseAll();
        HostPinnedMemoryManager::Default().ReleaseAll();
    }

    //////////////////////////////////////////////////////////////////////////
    static const char* MemGetErrorString(EMemStatus status)
    {
        switch (status) {
        case MEM_STATUS_SUCCESS: return "MEM_STATUS_SUCCESS";
        case MEM_STATUS_CUDA_ERROR: return "MEM_STATUS_CUDA_ERROR";
        case MEM_STATUS_INVALID_ARGUMENT: return "MEM_STATUS_INVALID_ARGUMENT";
        case MEM_STATUS_NOT_INITIALIZED: return "MEM_STATUS_NOT_INITIALIZED";
        case MEM_STATUS_OUT_OF_MEMORY: return "MEM_STATUS_OUT_OF_MEMORY";
        default: return "MEM_STATUS_UNKNOWN_ERROR";
        }
    }

    static bool ValidateArrayNotFreed(const float* ptr, size_t elements)
    {
        for (int i = 0; i < elements; ++i)
        {
            if (ptr[i] == 0xFEFEFEFE)
                return false;
        }

        return true;
    }

    static bool ValidateArrayModifiedAfterAlloc(const float* ptr, size_t elements)
    {
        for (int i = 0; i < elements; ++i)
        {
            if (ptr[i] == 0xABABABAB)
                return false;
        }

        return true;
    }
}
