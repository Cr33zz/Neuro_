#pragma once

#include <cuda.h>
#include <cudnn.h>
#include <cublas.h>

namespace Neuro
{
    void CudaAssert(cudaError_t code);
    void CudaAssert(cudnnStatus_t status);
    void CudaAssert(cublasStatus_t status);
    void CudaAssert(const char* error);
}

#ifndef NDEBUG
#   define CUDA_CHECK(op) CudaAssert(op)
#else
#   define CUDA_CHECK(op) op
#endif
