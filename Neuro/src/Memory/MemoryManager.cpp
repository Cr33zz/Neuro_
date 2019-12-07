#include <windows.h>
#include <debugapi.h>
#include <sstream>
#include <iomanip>
#include <cuda.h>

#include "Types.h"
#include "Tools.h"
#include "Memory/MemoryManager.h"
#include "Tensors/Cuda/CudaErrorCheck.h"

//#define ENABLE_MEMORY_LOGS

//#define MEMSET_ALLOCATED_MEMORY 0x00

#define DEVICE_ALLOC_GRANULARITY 512
#define HOST_ALLOC_GRANULARITY 256
#define DEVICE_NATIVE_GRANULARITY 512 * 1024
#define HOST_NATIVE_GRANULARITY 256 * 1024

#define MEM_CHECK(call) do { \
	EMemStatus status = (call); \
	if( status != MEM_STATUS_SUCCESS ) { \
        string msg = StringFormat("Memory manager failed with error: %s", MemGetErrorString(status)); \
        OutputDebugString(msg.c_str()); \
		NEURO_ASSERT(false, msg); \
		return status; \
	} \
} while(0)

#ifdef ENABLE_MEMORY_LOGS
#define MEM_DEBUG_INFO(info) do { stringstream ss; ss << info; OutputDebugString(ss.str().c_str()); } while(0)
#else
#define MEM_DEBUG_INFO(info) {}
#endif

namespace Neuro
{
    static string SizeToString(size_t size)
    {
        stringstream ss;
        ss << size;
        if (size > 1024)
        {
            if (size < 1024 * 1024)
                ss << "(~" << size / 1024 << "KB)";
            else
                ss << "(~" << size / (1024 * 1024) << "MB)";
        }
        return ss.str();
    }

    static inline size_t ceilInt(size_t m, size_t n)
    {
        return (m + n - 1) / n * n;
    }

    //////////////////////////////////////////////////////////////////////////
    MemoryManagerBase::MemoryManagerBase(size_t allocGranularity, size_t nativeAllocGranularity)
        : m_AllocGranularity(allocGranularity), m_NativeAllocGranularity(nativeAllocGranularity)
    {
    }

    //////////////////////////////////////////////////////////////////////////
    EMemStatus MemoryManagerBase::Allocate(void** ptr, size_t size, const string& annotation)
    {
        NVTXProfile p(__FUNCTION__, 0xFFFF0000);

        {
            unique_lock<mutex> deallocationsLocker(m_ScheduledFreeMtx);
#ifdef ENABLE_MEMORY_LOGS
            if (!m_ScheduledDeallocations.empty())
            {
                stringstream ss;
                ss << InternalName() << " releasing scheduled pointers..." << endl;
                OutputDebugString(ss.str().c_str());
            }
#endif
            if (!m_ScheduledDeallocations.empty())
            {
                // make a copy and release the lock to avoid dead-lock inside free
                auto scheduledDeallocsCopy = m_ScheduledDeallocations;
                m_ScheduledDeallocations.clear();
                deallocationsLocker.unlock();

                for (auto p : scheduledDeallocsCopy)
                    Free(p);
            }
        }

        unique_lock<mutex> allocFreeLocker(m_AllocFreeMtx);

        size = ceilInt(size, m_AllocGranularity);

        // Find the best fit.
        Block *best = nullptr, *prev = nullptr;
        MEM_CHECK(FindBestBlock(best, prev, size));

        // If there's no block left in the list of free blocks (with a sufficient size). Request a new block. 
        if (!best && !(m_Flags & MEM_FLAGS_CANNOT_GROW))
            MEM_CHECK(AllocateBlock(best, prev, size));

        // Make sure we do have a block or quit.
        if (!best)
        {
            *ptr = nullptr;
            DumpMemoryState("memory_manager.log");
            return MEM_STATUS_OUT_OF_MEMORY;
        }

        // Split the free block if needed.
        MEM_CHECK(SplitBlock(best, prev, size));

        // Push the node to the list of used nodes.
        best->SetNext(m_UsedBlocks);
        m_UsedBlocks = best;
        best->m_Annotation = annotation;

        m_AllocatedMemSize += size;
        m_AllocatedMemPeakSize = max(m_AllocatedMemSize, m_AllocatedMemPeakSize);

#ifdef ENABLE_MEMORY_LOGS
        stringstream ss;
        ss << InternalName() << " alloc '" << annotation << "' 0x" << hex << (__int64)m_UsedBlocks->GetData() << dec << " size " << SizeToString(size) << " total " << SizeToString(m_AllocatedMemSize) << " peak " << SizeToString(m_AllocatedMemPeakSize) << endl;
        OutputDebugString(ss.str().c_str());
#endif

        // Return the new pointer into memory.
        *ptr = m_UsedBlocks->GetData();

#ifdef MEMSET_ALLOCATED_MEMORY
        InternalMemset(m_UsedBlocks->GetData(), MEMSET_ALLOCATED_MEMORY, m_UsedBlocks->GetSize());
#endif
        return MEM_STATUS_SUCCESS;
    }

    EMemStatus MemoryManagerBase::ScheduleFree(void* ptr)
    {
        if (!ptr)
            return MEM_STATUS_SUCCESS;

        unique_lock<mutex> mtx(m_ScheduledFreeMtx);
        m_ScheduledDeallocations.push_back(ptr);

#ifdef ENABLE_MEMORY_LOGS
        stringstream ss;
        ss << InternalName() << " scheduled release 0x" << hex << (__int64)ptr << endl;
        OutputDebugString(ss.str().c_str());
#endif
        return MEM_STATUS_SUCCESS;
    }

    //////////////////////////////////////////////////////////////////////////
    EMemStatus MemoryManagerBase::Free(void* ptr)
    {
        NVTXProfile p(__FUNCTION__, 0xFFFF0000);

        if (!ptr)
            return MEM_STATUS_SUCCESS;

        unique_lock<mutex> allocFreeLocker(m_AllocFreeMtx);

        // Find the node in the list of used blocks.
        Block* curr = m_UsedBlocks, *prev = nullptr;
        for (; curr && curr->GetData() != ptr; curr = curr->GetNext())
            prev = curr;

        NEURO_ASSERT(curr, "Freeing unrecognized pointer");

        // Make sure we have found a node.
        if (!curr)
            return MEM_STATUS_INVALID_ARGUMENT;

        m_AllocatedMemSize -= curr->GetSize();

#ifdef ENABLE_MEMORY_LOGS
        stringstream ss;
        ss << InternalName() << " release '" << curr->m_Annotation << "' 0x" << hex << (__int64)ptr << dec << " size " << SizeToString(curr->GetSize()) << " total " << SizeToString(m_AllocatedMemSize) << endl;
        OutputDebugString(ss.str().c_str());
#endif

        // We have the node so release it.
        EMemStatus result = ReleaseBlock(curr, prev);        
        return result;
    }

    //////////////////////////////////////////////////////////////////////////
    EMemStatus MemoryManagerBase::ReleaseAll()
    {
        // Destroy used blocks. It's a kind of panic mode to avoid leaks. NOTE: Do that only with roots!!!
        while (m_UsedBlocks)
            MEM_CHECK(ReleaseBlock(m_UsedBlocks, nullptr));
        
        // We should be having only free blocks that are head blocks. Release those blocks.
        while (m_FreeBlocks)
        {
            Block* block = m_FreeBlocks;
            m_FreeBlocks = m_FreeBlocks->GetNext();
            delete block;
        }
        for (auto it = m_NativeBlocks.begin(); it != m_NativeBlocks.end(); it++)
        {
            void* data = it->ptr;
            InternalFree(data);
        }

        // We shouldn't have any used block left. Or, it means the user is causing memory leaks!
        return MEM_STATUS_SUCCESS;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////////
    EMemStatus MemoryManagerBase::ReleaseBlock(Block* curr, Block* prev)
    {
        // The current node cannot be NULL!
        NEURO_ASSERT(curr, "");

        // Change the connection of the node.
        if (prev)
            prev->SetNext(curr->GetNext());
        else
            m_UsedBlocks = curr->GetNext();

        // Find the location where this block should be added to the free list.
        prev = nullptr;
        Block* iter = m_FreeBlocks;
        for (; iter && iter->GetData() < curr->GetData(); iter = iter->GetNext())
            prev = iter;
        
        // Keep track of the successor of pred. We may lose track of it in the following "else".
        Block* next = prev ? prev->GetNext() : m_FreeBlocks;

        // We first check if we can merge the block with its predecessor in the list and curr can be merged.
        if (prev && prev->GetData() + prev->GetSize() == curr->GetData() && !curr->IsHead())
        {
            prev->SetSize(prev->GetSize() + curr->GetSize());
            delete curr;
            curr = prev;
        }
        else if (prev)
        {
            prev->SetNext(curr);
        }
        else
        {
            m_FreeBlocks = curr;
        }

        // Check if we can merge curr and next. We can't merge over native blocks boundaries.
        if (next && curr->GetData() + curr->GetSize() == next->GetData() && !next->IsHead())
        {
            curr->SetSize(curr->GetSize() + next->GetSize());
            curr->SetNext(next->GetNext());
            delete next;
        }
        else
        {
            curr->SetNext(next);
        }
        return MEM_STATUS_SUCCESS;
    }

    //////////////////////////////////////////////////////////////////////////
    EMemStatus MemoryManagerBase::AllocateBlock(Block*& curr, Block*& prev, size_t size)
    {
        curr = prev = nullptr;
        void* data = nullptr;
        
        size = ceilInt(size, m_NativeAllocGranularity);
        InternalAllocate(&data, size);
        MEM_DEBUG_INFO(">> returned address=0x" << hex << (size_t)data << "\n");
        AddNativeBlock(data, size);

        // If it failed, there's an unexpected issue.
        NEURO_ASSERT(data, "");

        // We have data, we now need to add it to the list of free nodes. We keep the list sorted.
        Block* next = m_FreeBlocks;
        for (; next && next->GetData() < data; next = next->GetNext())
            prev = next;
        
        curr = new Block((char*)data, size, next, true);
        if (!curr)
            return MEM_STATUS_OUT_OF_MEMORY;

        if (prev)
            prev->SetNext(curr);
        else
            m_FreeBlocks = curr;

        return MEM_STATUS_SUCCESS;
    }

    //////////////////////////////////////////////////////////////////////////
    EMemStatus MemoryManagerBase::SplitBlock(Block* curr, Block* prev, size_t size)
    {
        // We have two cases: 1/ It is the right size so we keep it or 2/ it is too large and we split the node.
        Block* next;
        if (curr->GetSize() == size)
        {
            next = curr->GetNext();
        }
        else
        {
            size_t remaining = curr->GetSize() - size;
            Block* newBlock = new Block(curr->GetData() + size, remaining, curr->GetNext(), false);
            if (!newBlock)
                return MEM_STATUS_OUT_OF_MEMORY;
            next = newBlock;
            curr->SetSize(size);
        }

        // Redo the "branching" in the nodes.
        if (prev)
            prev->SetNext(next);
        else
            m_FreeBlocks = next;
        return MEM_STATUS_SUCCESS;
    }

    //////////////////////////////////////////////////////////////////////////
    EMemStatus MemoryManagerBase::FindBestBlock(Block*& best, Block*& prev, size_t size)
    {
        best = prev = nullptr;
        for (Block* temp = m_FreeBlocks, *tempPrev = nullptr; temp; temp = temp->GetNext())
        {
            if (temp->GetSize() >= size && (!best || temp->GetSize() < best->GetSize()))
            {
                best = temp;
                prev = tempPrev;
            }
            tempPrev = temp;
        }
        return MEM_STATUS_SUCCESS;
    }

    //////////////////////////////////////////////////////////////////////////
    EMemStatus MemoryManagerBase::AddNativeBlock(void* ptr, size_t size)
    {
        m_NativeBlocks.push_back({ ptr, size });
        return MEM_STATUS_SUCCESS;
    }

    //////////////////////////////////////////////////////////////////////////
    EMemStatus MemoryManagerBase::RemoveNativeBlock(void *ptr)
    {
        bool found = false;
        for (auto it = m_NativeBlocks.begin(); it != m_NativeBlocks.end(); ++it)
        {
            if (it->ptr == ptr)
            {
                m_NativeBlocks.erase(it);
                found = true;
                break;
            }
        }

        if (!found)
            return MEM_STATUS_INVALID_ARGUMENT;
        else
            return MEM_STATUS_SUCCESS;
    }

    //////////////////////////////////////////////////////////////////////////
    EMemStatus MemoryManagerBase::GetUsedMemory(size_t& usedMemory) const
    {
        return GetMemory(usedMemory, m_UsedBlocks);
    }

    //////////////////////////////////////////////////////////////////////////
    EMemStatus MemoryManagerBase::GetFreeMemory(size_t& freeMemory) const
    {
        return GetMemory(freeMemory, m_FreeBlocks);
    }

    //////////////////////////////////////////////////////////////////////////
    EMemStatus MemoryManagerBase::GetMemory(size_t &size, const Block *head) const
    {
        size = 0;
        for (Block *curr = (Block*)head; curr; curr = curr->GetNext())
            size += curr->GetSize();
        return MEM_STATUS_SUCCESS;
    }

    //////////////////////////////////////////////////////////////////////////
    EMemStatus MemoryManagerBase::PrintList(FILE* file, const char* name, const Block* head) const
    {
        size_t size = 0;
        for (Block *curr = (Block*)head; curr; curr = curr->GetNext())
            size += curr->GetSize();
        
        fprintf(file, "| list=\"%s\", total=%s\n", name, SizeToString(size).c_str());
        for (Block *curr = (Block*)head; curr; curr = curr->GetNext())
            fprintf(file, "| | node=0x%016zx, data=0x%016zx, size=%zu, next=0x%016zx, head=%zu, annotation:'%s'\n", (size_t)curr, (size_t)curr->GetData(), (size_t)curr->GetSize(), (size_t)curr->GetNext(), (size_t)curr->IsHead(), curr->m_Annotation.c_str());
        fprintf(file, "|\n");
        return MEM_STATUS_SUCCESS;
    }

    //////////////////////////////////////////////////////////////////////////
    EMemStatus MemoryManagerBase::DumpMemoryState(const string& filename) const
    {
        auto file = fopen(filename.c_str(), "w");
        DumpMemoryState(file);
        fclose(file);
        return MEM_STATUS_SUCCESS;
    }

    //////////////////////////////////////////////////////////////////////////
    EMemStatus MemoryManagerBase::DumpMemoryState(FILE* file) const
    {
        size_t usedMemory, freeMemory;
        MEM_CHECK(GetUsedMemory(usedMemory));
        MEM_CHECK(GetFreeMemory(freeMemory));

        fprintf(file, "%s >>> used=%s, free=%s, peak=%s\n", InternalName(), SizeToString(usedMemory).c_str(), SizeToString(freeMemory).c_str(), SizeToString(m_AllocatedMemPeakSize).c_str());
        MEM_CHECK(PrintList(file, "used", m_UsedBlocks));
        MEM_CHECK(PrintList(file, "free", m_FreeBlocks));
        fprintf(file, "\n");
        return MEM_STATUS_SUCCESS;
    }

    //////////////////////////////////////////////////////////////////////////
    void MemoryManagerBase::UpdateAnnotation(void* ptr, const string& annotation)
    {
        if (!ptr)
            return;

        // device lookup
        Block *curr = m_UsedBlocks, *prev = nullptr;
        for (; curr && curr->GetData() != ptr; curr = curr->GetNext())
            prev = curr;

        if (curr)
        {
            curr->m_Annotation = annotation;
            return;
        }
    }

    //////////////////////////////////////////////////////////////////////////
    DeviceMemoryManager::DeviceMemoryManager()
        : MemoryManagerBase(DEVICE_ALLOC_GRANULARITY, DEVICE_NATIVE_GRANULARITY)
    {
        CUDA_CHECK(cudaStreamCreateWithFlags(&m_OffloadStream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&m_PreloadStream, cudaStreamNonBlocking));
        //CUDA_CHECK(cudaStreamCreate(&m_MemoryStream));
    }

    //////////////////////////////////////////////////////////////////////////
    DeviceMemoryManager& DeviceMemoryManager::Default()
    {
        static DeviceMemoryManager def;
        return def;
    }

    //////////////////////////////////////////////////////////////////////////
    void DeviceMemoryManager::InternalAllocate(void** ptr, size_t size, const string& annotation)
    {
        size_t freeBytes, totalBytes;
        cudaMemGetInfo(&freeBytes, &totalBytes);

        if (freeBytes < size)
            DumpMemoryState("device_out_of_memory.log");

        MEM_DEBUG_INFO("cudaMalloc(" << size << ")");
        CUDA_CHECK(cudaMalloc(ptr, size));
    }

    //////////////////////////////////////////////////////////////////////////
    void DeviceMemoryManager::InternalFree(void* ptr)
    {
        CUDA_CHECK(cudaFree(ptr));
    }

    //////////////////////////////////////////////////////////////////////////
    void DeviceMemoryManager::InternalMemset(void* ptr, uint8_t value, size_t size)
    {
        CUDA_CHECK(cudaMemset(ptr, value, size));
    }

    //////////////////////////////////////////////////////////////////////////
    EMemStatus DeviceMemoryManager::Reserve(size_t size)
    {
        Block *curr, *prev;
        MEM_CHECK(AllocateBlock(curr, prev, size));
        return MEM_STATUS_SUCCESS;
    }

    //////////////////////////////////////////////////////////////////////////
    EMemStatus DeviceMemoryManager::Offload(void* dst, void* src, size_t size, cudaEvent_t memEvent, cudaHostFn_t callback, void* userData)
    {
        NEURO_ASSERT(dst, "Host pinned memory is not allocated.");
        NEURO_ASSERT(cudaEventQuery(memEvent) == cudaSuccess, "Memory sync event is not ready.");
        CUDA_CHECK(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, m_OffloadStream));
        CUDA_CHECK(cudaEventRecord(memEvent, m_OffloadStream));
        if (callback)
            CUDA_CHECK(cudaLaunchHostFunc(m_OffloadStream, callback, userData));
        return MEM_STATUS_SUCCESS;
    }

    //////////////////////////////////////////////////////////////////////////
    EMemStatus DeviceMemoryManager::Preload(void* dst, void* src, size_t size, cudaEvent_t memEvent, cudaHostFn_t callback, void* userData)
    {
        NEURO_ASSERT(src, "Host pinned memory is not allocated.");
        NEURO_ASSERT(cudaEventQuery(memEvent) == cudaSuccess, "Memory sync event is not ready.");
        CUDA_CHECK(cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, m_PreloadStream));
        CUDA_CHECK(cudaEventRecord(memEvent, m_PreloadStream));
        if (callback)
            CUDA_CHECK(cudaLaunchHostFunc(m_PreloadStream, callback, userData));
        return MEM_STATUS_SUCCESS;
    }

    //////////////////////////////////////////////////////////////////////////
    EMemStatus DeviceMemoryManager::WaitForMemEvent(cudaEvent_t memEvent)
    {
        NVTXProfile p(__FUNCTION__, 0xFFFF0000);
        if (!memEvent)
            return MEM_STATUS_SUCCESS;

        CUDA_CHECK(cudaEventSynchronize(memEvent));
        return MEM_STATUS_SUCCESS;
    }

    //////////////////////////////////////////////////////////////////////////
    Neuro::EMemStatus DeviceMemoryManager::ForceOffloadStreamSync()
    {
        NVTXProfile p(__FUNCTION__, 0xFFFF0000);
        CUDA_CHECK(cudaStreamSynchronize(m_OffloadStream));
        return MEM_STATUS_SUCCESS;
    }

    //////////////////////////////////////////////////////////////////////////
    Neuro::EMemStatus DeviceMemoryManager::ForcePreloadStreamSync()
    {
        NVTXProfile p(__FUNCTION__, 0xFFFF0000);
        CUDA_CHECK(cudaStreamSynchronize(m_PreloadStream));
        return MEM_STATUS_SUCCESS;
    }

    //////////////////////////////////////////////////////////////////////////
    HostMemoryManager::HostMemoryManager()
        : MemoryManagerBase(HOST_ALLOC_GRANULARITY, HOST_NATIVE_GRANULARITY)
    {
    }

    //////////////////////////////////////////////////////////////////////////
    HostMemoryManager& HostMemoryManager::Default()
    {
        static HostMemoryManager def;
        return def;
    }

    //////////////////////////////////////////////////////////////////////////
    void HostMemoryManager::InternalAllocate(void** ptr, size_t size, const string& annotation)
    {
        MEM_DEBUG_INFO("malloc(" << size << ")");
        *ptr = malloc(size);
    }

    //////////////////////////////////////////////////////////////////////////
    void HostMemoryManager::InternalFree(void* ptr)
    {
        free(ptr);
    }

    //////////////////////////////////////////////////////////////////////////
    void HostMemoryManager::InternalMemset(void* ptr, uint8_t value, size_t size)
    {
        memset(ptr, value, size);
    }

    //////////////////////////////////////////////////////////////////////////
    HostPinnedMemoryManager::HostPinnedMemoryManager()
        : MemoryManagerBase(HOST_ALLOC_GRANULARITY, HOST_NATIVE_GRANULARITY)
    {
    }

    //////////////////////////////////////////////////////////////////////////
    HostPinnedMemoryManager& HostPinnedMemoryManager::Default()
    {
        static HostPinnedMemoryManager def;
        return def;
    }

    //////////////////////////////////////////////////////////////////////////
    void HostPinnedMemoryManager::InternalAllocate(void** ptr, size_t size, const string& annotation)
    {
        MEM_DEBUG_INFO("cudaMallocHost(" << size << ")");
        CUDA_CHECK(cudaMallocHost(ptr, size));
    }

    //////////////////////////////////////////////////////////////////////////
    void HostPinnedMemoryManager::InternalFree(void* ptr)
    {
        CUDA_CHECK(cudaFreeHost(ptr));
    }

    //////////////////////////////////////////////////////////////////////////
    void HostPinnedMemoryManager::InternalMemset(void* ptr, uint8_t value, size_t size)
    {
        memset(ptr, value, size);
    }
}