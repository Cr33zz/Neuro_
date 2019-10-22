#include <windows.h>
#include <debugapi.h>
#include <sstream>
#include <iomanip>
#include <cuda.h>

#include "Types.h"
#include "Tools.h"
#include "Memory/MemoryManager.h"
#include "Tensors/Cuda/CudaErrorCheck.h"

#define ENABLE_MEMORY_LOGS

#define DEVICE_ALLOC_GRANULARITY 512
#define CUDA_GRANULARITY 128 * 1024

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
    static void PrintSize(stringstream& ss, size_t size)
    {
        ss << size << "B";
        if (size > 1024)
        {
            if (size < 1024 * 1024)
                ss << "(~" << size / 1024 << "KB)";
            else
                ss << "(~" << size / (1024 * 1024) << "MB)";
        }
    }

    //////////////////////////////////////////////////////////////////////////
    MemoryManager::MemoryManager()
    {
        cudaStreamCreate(&m_MemoryStream);
    }

    //////////////////////////////////////////////////////////////////////////
    MemoryManager& MemoryManager::Default()
    {
        static MemoryManager def;
        return def;
    }

    static inline size_t ceilInt(size_t m, size_t n)
    {
        return (m + n - 1) / n * n;
    }

    //////////////////////////////////////////////////////////////////////////
    EMemStatus MemoryManager::AllocateDevice(void** ptr, size_t size, const string& annotation)
    {
        size = ceilInt(size, DEVICE_ALLOC_GRANULARITY);

        // Find the best fit.
        Block *best = nullptr, *prev = nullptr;
        MEM_CHECK(FindBestBlockUnsafe(best, prev, size));

        // If there's no block left in the list of free blocks (with a sufficient size). Request a new block. 
        if (!best && !(m_Flags & MEM_FLAGS_CANNOT_GROW))
            MEM_CHECK(AllocateBlockUnsafe(best, prev, size));

        // Make sure we do have a block or quit.
        if (!best)
        {
            *ptr = nullptr;
            PrintMemoryState("memory_manager.log");
            return MEM_STATUS_OUT_OF_MEMORY;
        }

        // Split the free block if needed.
        MEM_CHECK(ExtractBlockUnsafe(best, prev, size, false));

        // Push the node to the list of used nodes.
        best->SetNext(m_UsedBlocks);
        m_UsedBlocks = best;
        best->m_Annotation = annotation;

        m_AllocatedDeviceMemSize += size;
        m_AllocatedDeviceMemSizePeak = max(m_AllocatedDeviceMemSize, m_AllocatedDeviceMemSizePeak);

#ifdef ENABLE_MEMORY_LOGS
        stringstream ss;
        ss << "Device alloc '" << annotation << "' 0x" << hex << (__int64)m_UsedBlocks->GetData() << dec << " size ";
        PrintSize(ss, size);
        ss << " total ";
        PrintSize(ss, m_AllocatedDeviceMemSize);
        ss << " peak ";
        PrintSize(ss, m_AllocatedDeviceMemSizePeak);
        ss << endl;
        OutputDebugString(ss.str().c_str());
#endif

        // Return the new pointer into memory.
        *ptr = m_UsedBlocks->GetData();
        return MEM_STATUS_SUCCESS;
    }

    //////////////////////////////////////////////////////////////////////////
    EMemStatus MemoryManager::ReleaseDevice(void* ptr)
    {
        if (!ptr)
            return MEM_STATUS_SUCCESS;

        // Find the node in the list of used blocks.
        Block *curr = m_UsedBlocks, *prev = nullptr;
        for (; curr && curr->GetData() != ptr; curr = curr->GetNext())
            prev = curr;

        // Make sure we have found a node.
        if (!curr)
            return MEM_STATUS_INVALID_ARGUMENT;

        m_AllocatedDeviceMemSize -= curr->GetSize();

#ifdef ENABLE_MEMORY_LOGS
        stringstream ss;
        ss << "Device release '" << curr->m_Annotation << "' 0x" << hex << (__int64)ptr << dec << " size ";
        PrintSize(ss, curr->GetSize());
        ss << " total ";
        PrintSize(ss, m_AllocatedDeviceMemSize);
        ss << endl;
        OutputDebugString(ss.str().c_str());
#endif

        // We have the node so release it.
        EMemStatus result = ReleaseBlockUnsafe(curr, prev);        
        return result;
    }

    //////////////////////////////////////////////////////////////////////////
    EMemStatus MemoryManager::AllocateHostPinned(void** ptr, size_t size, const string& annotation)
    {
        CUDA_CHECK(cudaMallocHost(ptr, size));
        m_AllocatedHostPinnedMemSize += size;
        m_AllocatedHostPinnedMemPeakSize = max(m_AllocatedHostPinnedMemSize, m_AllocatedHostPinnedMemPeakSize);
        m_HostPinnedAllocations.push_back({ *ptr, size, annotation});

#ifdef ENABLE_MEMORY_LOGS
        stringstream ss;
        ss << "Host pinned alloc '" << annotation << "' 0x" << hex << (__int64)*ptr << dec << " size ";
        PrintSize(ss, size);
        ss << " total ";
        PrintSize(ss, m_AllocatedHostPinnedMemSize);
        ss << " peak ";
        PrintSize(ss, m_AllocatedHostPinnedMemPeakSize);
        ss << endl;
        OutputDebugString(ss.str().c_str());
#endif

        if (ptr)
            return MEM_STATUS_SUCCESS;
        return MEM_STATUS_OUT_OF_MEMORY;
    }

    //////////////////////////////////////////////////////////////////////////
    EMemStatus MemoryManager::ReleaseHostPinned(void* ptr)
    {
        if (!ptr)
            return MEM_STATUS_SUCCESS;
        auto it = find_if(m_HostPinnedAllocations.begin(), m_HostPinnedAllocations.end(), [&](const HostAlloc& a) { return a.ptr == ptr; });
        NEURO_ASSERT(it != m_HostPinnedAllocations.end(), "");
        m_AllocatedHostPinnedMemSize -= it->size;

#ifdef ENABLE_MEMORY_LOGS
        stringstream ss;
        ss << "Host pinned release '" << it->annotation << "' 0x" << hex << (__int64)ptr << dec << " size ";
        PrintSize(ss, it->size);
        ss << " total ";
        PrintSize(ss, m_AllocatedHostPinnedMemSize);
        ss << endl;
        OutputDebugString(ss.str().c_str());
#endif

        m_HostPinnedAllocations.erase(it);
        CUDA_CHECK(cudaFreeHost(ptr));
        return MEM_STATUS_SUCCESS;
    }

    //////////////////////////////////////////////////////////////////////////
    EMemStatus MemoryManager::AllocateHost(void** ptr, size_t size, const string& annotation)
    {
        *ptr = malloc(size);
        m_AllocatedHostMemSize += size;
        m_AllocatedHostMemSizePeak = max(m_AllocatedHostMemSize, m_AllocatedHostMemSizePeak);
        m_HostAllocations.push_back({ *ptr, size, annotation });

#ifdef ENABLE_MEMORY_LOGS
        stringstream ss;
        ss << "Host alloc '" << annotation << "' 0x" << hex << (__int64)*ptr << dec << " size ";
        PrintSize(ss, size);
        ss << " total ";
        PrintSize(ss, m_AllocatedHostMemSize);
        ss << " peak ";
        PrintSize(ss, m_AllocatedHostMemSizePeak);
        ss << endl;
        OutputDebugString(ss.str().c_str());
#endif

        if (ptr)
            return MEM_STATUS_SUCCESS;
        return MEM_STATUS_OUT_OF_MEMORY;
    }

    //////////////////////////////////////////////////////////////////////////
    EMemStatus MemoryManager::ReleaseHost(void* ptr)
    {
        if (!ptr)
            return MEM_STATUS_SUCCESS;

        auto it = find_if(m_HostAllocations.begin(), m_HostAllocations.end(), [&](const HostAlloc& a) { return a.ptr == ptr; });
        NEURO_ASSERT(it != m_HostAllocations.end(), "");
        m_AllocatedHostMemSize -= it->size;

#ifdef ENABLE_MEMORY_LOGS
        stringstream ss;
        ss << "Host release '" << it->annotation << "' 0x" << hex << (__int64)ptr << dec << " size ";
        PrintSize(ss, it->size);
        ss << " total ";
        PrintSize(ss, m_AllocatedHostMemSize);
        ss << endl;
        OutputDebugString(ss.str().c_str());
#endif

        m_HostAllocations.erase(it);
        free(ptr);
        return MEM_STATUS_SUCCESS;
    }

    //////////////////////////////////////////////////////////////////////////
    EMemStatus MemoryManager::ReleaseAllUnsafe()
    {
        // Destroy used blocks. It's a kind of panic mode to avoid leaks. NOTE: Do that only with roots!!!
        while (m_UsedBlocks)
            MEM_CHECK(ReleaseBlockUnsafe(m_UsedBlocks, nullptr));
        
        // We should be having only free blocks that are head blocks. Release those blocks.
        while (m_FreeBlocks)
        {
            Block *block = m_FreeBlocks;
            m_FreeBlocks = m_FreeBlocks->GetNext();
            delete block;
        }
        for (auto it = m_CudaBlocks.begin(); it != m_CudaBlocks.end(); it++)
        {
            void *data = it->ptr;
            CUDA_CHECK(cudaFree(data));
        }

        // We shouldn't have any used block left. Or, it means the user is causing memory leaks!
        return MEM_STATUS_SUCCESS;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////////
    EMemStatus MemoryManager::ReleaseBlockUnsafe(Block* curr, Block* prev)
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

        // Check if we can merge curr and next. We can't merge over "cudaMalloc" boundaries.
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
    EMemStatus MemoryManager::AllocateBlockUnsafe(Block *&curr, Block *&prev, size_t size)
    {
        // Reset the outputs.
        curr = prev = nullptr;
        // Try to allocate data from the parent or the device.
        void *data = nullptr;
        Block *last = m_FreeBlocks, *last_prev = nullptr;
        if (last)
        {
            for (; last->GetNext(); last = last->GetNext())
                last_prev = last;
            
            // test run - need to change if cudaMalloc actually returns discontinuous address
            size_t extra_size = size - last->GetSize();
            extra_size = ceilInt(extra_size, CUDA_GRANULARITY);

            MEM_DEBUG_INFO("cudaMalloc(" << extra_size << ")");
            CUDA_CHECK(cudaMalloc(&data, extra_size));
            MEM_DEBUG_INFO(">> returned address=0x" << hex << (size_t)data << "\n");

            NEURO_ASSERT(data == last->GetData() + last->GetSize(), "Discontinuous CUDA memory allocation detected.");

            AddCudaBlockUnsafe(data, extra_size);

            if (last->GetData() + last->GetSize() == (char*)data)
            {
                last->SetSize(last->GetSize() + extra_size);
                curr = last;
                prev = last_prev;
                return MEM_STATUS_SUCCESS;
            }
            else
            {
                MEM_DEBUG_INFO("cudaFree(" << extra_size << ", 0x" << hex << (size_t)data << ")");
                CUDA_CHECK(cudaFree(data));
                MEM_DEBUG_INFO(">> success\n");
                RemoveCudaBlockUnsafe(data);
                data = NULL;
            }

        }

        // if last == NULL or cudaMalloc cannot be merged
        size = ceilInt(size, CUDA_GRANULARITY);
        MEM_DEBUG_INFO("cudaMalloc(" << size << ")");
        CUDA_CHECK(cudaMalloc(&data, size));
        MEM_DEBUG_INFO(">> returned address=0x" << hex << (size_t)data << "\n");
        AddCudaBlockUnsafe(data, size);

        // If it failed, there's an unexpected issue.
        NEURO_ASSERT(data, "");

        // We have data, we now need to add it to the list of free nodes. We keep the list sorted.
        Block *next = m_FreeBlocks;
        for (; next && next->GetData() < data; next = next->GetNext())
            prev = next;
        
        curr = new Block((char*)data, size, next, false);
        if (!curr)
            return MEM_STATUS_OUT_OF_MEMORY;

        if (prev)
            prev->SetNext(curr);
        else
            m_FreeBlocks = curr;

        return MEM_STATUS_SUCCESS;
    }

    //////////////////////////////////////////////////////////////////////////
    EMemStatus MemoryManager::ExtractBlockUnsafe(Block *curr, Block *prev, size_t size, bool stolen)
    {
        // We have two cases: 1/ It is the right size so we keep it or 2/ it is too large and we split the node.
        Block *next;
        if (curr->GetSize() == size)
        {
            next = curr->GetNext();
        }
        else
        {
            size_t remaining = curr->GetSize() - size;
            Block *newBlock = new Block(curr->GetData() + size, remaining, curr->GetNext(), stolen);
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
    EMemStatus MemoryManager::FindBestBlockUnsafe(Block*& best, Block*& prev, size_t size)
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
    EMemStatus MemoryManager::Reserve(size_t size)
    {
        Block *curr, *prev;
        MEM_CHECK(AllocateBlockUnsafe(curr, prev, size));
        m_Size = size;
        return MEM_STATUS_SUCCESS;
    }

    //////////////////////////////////////////////////////////////////////////
    EMemStatus MemoryManager::Offload(void* dst, void* src, size_t size, cudaEvent_t memEvent)
    {
        NEURO_ASSERT(dst, "Host pinned memory is not allocated.");
        NEURO_ASSERT(cudaEventQuery(memEvent) == cudaSuccess, "Memory sync event is not ready.");
        CUDA_CHECK(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, m_MemoryStream));
        CUDA_CHECK(cudaEventRecord(memEvent, m_MemoryStream));
        return MEM_STATUS_SUCCESS;
    }

    //////////////////////////////////////////////////////////////////////////
    EMemStatus MemoryManager::Prefetch(void* dst, void* src, size_t size, cudaEvent_t memEvent)
    {
        NEURO_ASSERT(src, "Host pinned memory is not allocated.");
        NEURO_ASSERT(cudaEventQuery(memEvent) == cudaSuccess, "Memory sync event is not ready.");
        CUDA_CHECK(cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, m_MemoryStream));
        CUDA_CHECK(cudaEventRecord(memEvent, m_MemoryStream));
        return MEM_STATUS_SUCCESS;
    }

    //////////////////////////////////////////////////////////////////////////
    Neuro::EMemStatus MemoryManager::WaitForMemEvent(cudaEvent_t memEvent)
    {
        if (!memEvent)
            return MEM_STATUS_SUCCESS;

        CUDA_CHECK(cudaEventSynchronize(memEvent));
        return MEM_STATUS_SUCCESS;
    }

    //////////////////////////////////////////////////////////////////////////
    EMemStatus MemoryManager::AddCudaBlockUnsafe(void* ptr, size_t size)
    {
        m_CudaBlocks.push_back({ ptr, size });
        return MEM_STATUS_SUCCESS;
    }

    //////////////////////////////////////////////////////////////////////////
    EMemStatus MemoryManager::RemoveCudaBlockUnsafe(void *ptr)
    {
        bool found = false;
        for (auto it = m_CudaBlocks.begin(); it != m_CudaBlocks.end(); ++it)
        {
            if (it->ptr == ptr)
            {
                m_CudaBlocks.erase(it);
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
    EMemStatus MemoryManager::GetUsedMemoryUnsafe(size_t& usedMemory) const
    {
        return GetMemoryUnsafe(usedMemory, m_UsedBlocks);
    }

    //////////////////////////////////////////////////////////////////////////
    EMemStatus MemoryManager::GetFreeMemoryUnsafe(size_t& freeMemory) const
    {
        return GetMemoryUnsafe(freeMemory, m_FreeBlocks);
    }

    //////////////////////////////////////////////////////////////////////////
    EMemStatus MemoryManager::GetMemoryUnsafe(size_t &size, const Block *head) const
    {
        size = 0;
        for (Block *curr = (Block*)head; curr; curr = curr->GetNext())
            size += curr->GetSize();
        return MEM_STATUS_SUCCESS;
    }

    //////////////////////////////////////////////////////////////////////////
    EMemStatus MemoryManager::PrintListUnsafe(FILE* file, const char* name, const Block* head) const
    {
        size_t size = 0;
        for (Block *curr = (Block*)head; curr; curr = curr->GetNext())
            size += curr->GetSize();
        
        fprintf(file, "| list=\"%s\", size=%zu\n", name, size);
        for (Block *curr = (Block*)head; curr; curr = curr->GetNext())
            fprintf(file, "| | node=0x%016zx, data=0x%016zx, size=%zu, next=0x%016zx, head=%2zu, annotation:'%s'\n", (size_t)curr, (size_t)curr->GetData(), (size_t)curr->GetSize(), (size_t)curr->GetNext(), (size_t)curr->IsHead(), curr->m_Annotation.c_str());
        fprintf(file, "|\n");
        return MEM_STATUS_SUCCESS;
    }

    //////////////////////////////////////////////////////////////////////////
    EMemStatus MemoryManager::PrintHostAllocs(FILE* file, const char* name, const list<HostAlloc>& allocs) const
    {
        fprintf(file, "| list=\"%s\", size=%zu\n", name, m_AllocatedHostPinnedMemSize);
        for (auto& a : allocs)
            fprintf(file, "| | data=0x%016zx, size=%zu, annotation:'%s'\n", (size_t)a.ptr, (size_t)a.size, a.annotation.c_str());
        fprintf(file, "|\n");
        return MEM_STATUS_SUCCESS;
    }

    //////////////////////////////////////////////////////////////////////////
    EMemStatus MemoryManager::PrintMemoryState(const string& filename) const
    {
        auto file = fopen(filename.c_str(), "w");
        size_t streamCode = (size_t)m_MemoryStream;
        size_t usedMemory, freeMemory;
        MEM_CHECK(GetUsedMemoryUnsafe(usedMemory));
        MEM_CHECK(GetFreeMemoryUnsafe(freeMemory));

        fprintf(file, ">> stream=0x%016zx, used=%zuKB, free=%zuKB, peak=%zuKB, pinned=%zuKB\n", streamCode, usedMemory / 1024, freeMemory / 1024, m_AllocatedDeviceMemSizePeak / 1024, m_AllocatedHostPinnedMemSize / 1024);
        MEM_CHECK(PrintListUnsafe(file, "device_used", m_UsedBlocks));
        MEM_CHECK(PrintListUnsafe(file, "device_free", m_FreeBlocks));
        MEM_CHECK(PrintHostAllocs(file, "host", m_HostAllocations));
        MEM_CHECK(PrintHostAllocs(file, "host_pinned", m_HostPinnedAllocations));
        fprintf(file, "\n");
        fclose(file);

        return MEM_STATUS_SUCCESS;
    }
}