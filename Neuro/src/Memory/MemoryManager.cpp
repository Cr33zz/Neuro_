#include <windows.h>
#include <debugapi.h>
#include <sstream>
#include <iomanip>
#include <cuda.h>

#include "Types.h"
#include "Tools.h"
#include "Memory/MemoryManager.h"
#include "Tensors/Cuda/CudaErrorCheck.h"

#define ENABLE_GPU_MEMORY_LOGS

#define MEM_GRANULARITY 512
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

#if 1
#define MEM_DEBUG_INFO(info) do { stringstream ss; ss << info; OutputDebugString(ss.str().c_str()); } while(0)
#else
#define MEM_DEBUG_INFO(info)
#endif

//https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__UNIFIED.html
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
    EMemStatus MemoryManager::Allocate(void** ptr, size_t size, bool isBlocking)
    {
        size = ceilInt(size, MEM_GRANULARITY);

        // If the client is not blocking, we have to explicitly synchronize before giving one buffer.
        if (!isBlocking)
            CUDA_CHECK(cudaStreamSynchronize(m_MemoryStream));

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
            return MEM_STATUS_OUT_OF_MEMORY;
        }

        // Split the free block if needed.
        MEM_CHECK(ExtractBlockUnsafe(best, prev, size, false));

        // Push the node to the list of used nodes.
        best->SetNext(m_UsedBlocks);
        m_UsedBlocks = best;

        m_AllocatedMemSize += size;

#ifdef ENABLE_GPU_MEMORY_LOGS
        stringstream ss;
        ss << "Alloc 0x" << hex << (__int64)m_UsedBlocks->GetData() << dec << " size ";
        PrintSize(ss, size);
        ss << " total_allocated ";
        PrintSize(ss, m_AllocatedMemSize);
        ss << endl;
        OutputDebugString(ss.str().c_str());
#endif

        // Return the new pointer into memory.
        *ptr = m_UsedBlocks->GetData();
        return MEM_STATUS_SUCCESS;
    }

    //////////////////////////////////////////////////////////////////////////
    EMemStatus MemoryManager::Release(void* ptr)
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

        m_AllocatedMemSize -= curr->GetSize();

#ifdef ENABLE_GPU_MEMORY_LOGS
        stringstream ss;
        ss << "Release 0x" << hex << (__int64)ptr << dec << " size ";
        PrintSize(ss, curr->GetSize());
        ss << " total_allocated ";
        PrintSize(ss, m_AllocatedMemSize);
        ss << endl;
        OutputDebugString(ss.str().c_str());
#endif

        // We have the node so release it.
        EMemStatus result = ReleaseBlockUnsafe(curr, prev);        
        return result;
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
    EMemStatus MemoryManager::AllocateForOffload(void** ptr, size_t size)
    {
        CUDA_CHECK(cudaMallocHost(ptr, size));
        if (ptr)
            return MEM_STATUS_SUCCESS;
        return MEM_STATUS_OUT_OF_MEMORY;
    }

    //////////////////////////////////////////////////////////////////////////
    EMemStatus MemoryManager::ReleaseForOffload(void* ptr)
    {
        if (!ptr)
            return MEM_STATUS_SUCCESS;

        CUDA_CHECK(cudaFreeHost(ptr));
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
}