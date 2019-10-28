#include <string>
#include <sstream>
#include <windows.h>
#include <debugapi.h>
#include <assert.h>

#include "Types.h"
#include "Tensors/Cuda/CudaErrorCheck.h"

namespace Neuro
{
    using namespace std;

    //////////////////////////////////////////////////////////////////////////
    void CudaAssert(cudaError_t code)
    {
        if (code != cudaSuccess)
            CudaAssert(cudaGetErrorString(code));
    }

    //////////////////////////////////////////////////////////////////////////
    void CudaAssert(cublasStatus_t status)
    {
        if (status != CUBLAS_STATUS_SUCCESS)
        {
            string error = to_string(status);
            switch (status)
            {
            case CUBLAS_STATUS_SUCCESS:
                error = "CUBLAS_STATUS_SUCCESS";
            case CUBLAS_STATUS_NOT_INITIALIZED:
                error = "CUBLAS_STATUS_NOT_INITIALIZED";
            case CUBLAS_STATUS_ALLOC_FAILED:
                error = "CUBLAS_STATUS_ALLOC_FAILED";
            case CUBLAS_STATUS_INVALID_VALUE:
                error = "CUBLAS_STATUS_INVALID_VALUE";
            case CUBLAS_STATUS_ARCH_MISMATCH:
                error = "CUBLAS_STATUS_ARCH_MISMATCH";
            case CUBLAS_STATUS_MAPPING_ERROR:
                error = "CUBLAS_STATUS_MAPPING_ERROR";
            case CUBLAS_STATUS_EXECUTION_FAILED:
                error = "CUBLAS_STATUS_EXECUTION_FAILED";
            case CUBLAS_STATUS_INTERNAL_ERROR:
                error = "CUBLAS_STATUS_INTERNAL_ERROR";
            }

            CudaAssert(error.c_str());
        }
    }

    //////////////////////////////////////////////////////////////////////////
    void CudaAssert(cudnnStatus_t status)
    {
        if (status != CUDNN_STATUS_SUCCESS)
            CudaAssert(cudnnGetErrorString(status));
    }

    //////////////////////////////////////////////////////////////////////////
    void CudaAssert(EMemStatus status)
    {
        if (status != MEM_STATUS_SUCCESS)
            CudaAssert(MemGetErrorString(status));
    }

    //////////////////////////////////////////////////////////////////////////
    void CudaAssert(const char* error)
    {
        stringstream ss;
        ss << "CUDA error: " << error << endl;
        OutputDebugString(ss.str().c_str());
        NEURO_ASSERT(false, ss.str().c_str());
    }
}