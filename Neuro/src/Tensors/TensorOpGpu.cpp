#include <sstream>
#include <windows.h>
#include <debugapi.h>

#include "Tools.h"
#include "Tensors/TensorOpGpu.h"
#include "Tensors/Cuda/CudaErrorCheck.h"
#include "Tensors/Cuda/CudaKernels.h"
#include "Memory/MemoryManager.h"

namespace Neuro
{
    bool TensorOpGpu::s_Initialized = false;
    cudaDeviceProp TensorOpGpu::s_CudaDevProp;
    cublasHandle_t TensorOpGpu::s_CublasHandle = nullptr;
    cudnnHandle_t TensorOpGpu::s_CudnnHandle = nullptr;

    static const int INNER_KERNEL_LOOP_LENGTH = 64; // for simple per-element kernels

    //////////////////////////////////////////////////////////////////////////
    TensorOpGpu::TensorOpGpu()
    {
        if (!s_Initialized)
        {
            s_Initialized = true;

            int cudaDevicesNum;
            cudaGetDeviceCount(&cudaDevicesNum);

            if (cudaDevicesNum > 0)
            {
                cublasCreate_v2(&s_CublasHandle);
                cudnnCreate(&s_CudnnHandle);
                cudaGetDeviceProperties(&s_CudaDevProp, 0);

                //cudnnSetCallback(CUDNN_SEV_INFO_EN, nullptr, CudnnLog);
                size_t freeBytes, totalBytes;
                cudaMemGetInfo(&freeBytes, &totalBytes);

                stringstream ss;
                ss << "GPU >> " << s_CudaDevProp.name << " tpb=" << s_CudaDevProp.maxThreadsPerBlock << " SM_count=" << s_CudaDevProp.multiProcessorCount << " available/total_memory=" << freeBytes/(1024*1024) << "/" << totalBytes/(1024*1024) << "MB\n";
                OutputDebugString(ss.str().c_str());
            }
        }
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::DeallocateWorkspace(void* ptr)
    {
        CUDA_CHECK(DeviceMemoryManager::Default().ScheduleFree(ptr));
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::Zero(Tensor& input) const
    {
        NVTXProfile nvtxProfile(__FUNCTION__, 0xFF004A7F);
        input.OverrideDevice();
        CUDA_CHECK(cudaMemset(input.GetDevicePtr(), 0, input.Length() * sizeof(float)));
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::One(Tensor& input) const
    {
        NVTXProfile nvtxProfile(__FUNCTION__, 0xFF004A7F);
        float one = 1.f;
        unsigned int* oneBits = reinterpret_cast<unsigned int*>(&one);
        input.OverrideDevice();
        cuMemsetD32(CUdeviceptr(input.GetDevicePtr()), *oneBits, input.Length());
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::Add(float alpha, const Tensor& t1, float beta, const Tensor& t2, Tensor& output) const
    {
        NVTXProfile nvtxProfile(__FUNCTION__, 0xFF004A7F);
        t1.CopyToDevice();
        t2.CopyToDevice();
        output.OverrideDevice();

        if (!t1.SameDimensionsExceptBatches(t2))
        {
            if (t1.SameDimensionsExceptBatches(output))
            {
                if (t2.SameDimensionsOrOne(output))
                {
                    cudnnOpTensorDescriptor_t addDesc; cudnnCreateOpTensorDescriptor(&addDesc);
                    cudnnTensorDescriptor_t t1Desc; cudnnCreateTensorDescriptor(&t1Desc);
                    cudnnTensorDescriptor_t t2Desc; cudnnCreateTensorDescriptor(&t2Desc);
                    cudnnTensorDescriptor_t outputDesc; cudnnCreateTensorDescriptor(&outputDesc);

                    cudnnSetOpTensorDescriptor(addDesc, CUDNN_OP_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN);
                    cudnnSetTensor4dDescriptor(t1Desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, t1.GetShape().Dimensions[3], t1.GetShape().Dimensions[2], t1.GetShape().Dimensions[1], t1.GetShape().Dimensions[0]);
                    cudnnSetTensor4dDescriptor(t2Desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, t2.GetShape().Dimensions[3], t2.GetShape().Dimensions[2], t2.GetShape().Dimensions[1], t2.GetShape().Dimensions[0]);
                    cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, output.GetShape().Dimensions[3], output.GetShape().Dimensions[2], output.GetShape().Dimensions[1], output.GetShape().Dimensions[0]);

                    float beta2 = 0.f;
                    CUDA_CHECK(cudnnOpTensor(s_CudnnHandle, addDesc, &alpha, t1Desc, t1.GetDevicePtr(), &beta, t2Desc, t2.GetDevicePtr(), &beta2, outputDesc, output.GetDevicePtr()));
                    cudaStreamSynchronize(0);
                    return;
                }

                dim3 blocks, threads;
                GetKernelRunParams(output.Length(), blocks, threads, s_CudaDevProp.maxThreadsPerBlock);
                CudaKernels::AddBroadcast(
                    blocks,
                    threads,
                    alpha,
                    t1.GetDevicePtr(),
                    beta,
                    t2.GetDevicePtr(),
                    t2.Width(),
                    t2.Height(),
                    t2.Depth(),
                    t2.Batch(),
                    output.GetDevicePtr(),
                    output.Width(),
                    output.Height(),
                    output.Depth(),
                    output.Batch());
                cudaStreamSynchronize(0);
                return;
            }
            else if (t2.SameDimensionsExceptBatches(output))
            {
                if (t1.SameDimensionsOrOne(output))
                {
                    cudnnOpTensorDescriptor_t addDesc; cudnnCreateOpTensorDescriptor(&addDesc);
                    cudnnTensorDescriptor_t t1Desc; cudnnCreateTensorDescriptor(&t1Desc);
                    cudnnTensorDescriptor_t t2Desc; cudnnCreateTensorDescriptor(&t2Desc);
                    cudnnTensorDescriptor_t outputDesc; cudnnCreateTensorDescriptor(&outputDesc);

                    cudnnSetOpTensorDescriptor(addDesc, CUDNN_OP_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN);
                    cudnnSetTensor4dDescriptor(t1Desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, t1.GetShape().Dimensions[3], t1.GetShape().Dimensions[2], t1.GetShape().Dimensions[1], t1.GetShape().Dimensions[0]);
                    cudnnSetTensor4dDescriptor(t2Desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, t2.GetShape().Dimensions[3], t2.GetShape().Dimensions[2], t2.GetShape().Dimensions[1], t2.GetShape().Dimensions[0]);
                    cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, output.GetShape().Dimensions[3], output.GetShape().Dimensions[2], output.GetShape().Dimensions[1], output.GetShape().Dimensions[0]);

                    float beta2 = 0.f;
                    CUDA_CHECK(cudnnOpTensor(s_CudnnHandle, addDesc, &beta, t2Desc, t2.GetDevicePtr(), &alpha, t1Desc, t1.GetDevicePtr(), &beta2, outputDesc, output.GetDevicePtr()));
                    cudaStreamSynchronize(0);
                    return;
                }

                dim3 blocks, threads;
                GetKernelRunParams(output.Length(), blocks, threads, s_CudaDevProp.maxThreadsPerBlock);
                CudaKernels::AddBroadcast(
                    blocks,
                    threads,
                    beta,
                    t2.GetDevicePtr(),
                    alpha,
                    t1.GetDevicePtr(),
                    t1.Width(),
                    t1.Height(),
                    t1.Depth(),
                    t1.Batch(),
                    output.GetDevicePtr(),
                    output.Width(),
                    output.Height(),
                    output.Depth(),
                    output.Batch());
                cudaStreamSynchronize(0);
                return;
            }
            else
                __super::Add(alpha, t1, beta, t2, output);
        }

        if (t2.Batch() == t1.Batch())
        {
            CUDA_CHECK(cublasSgeam(
                s_CublasHandle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                t1.Length(),
                1,
                &alpha,
                t1.GetDevicePtr(),
                t1.Length(),
                &beta,
                t2.GetDevicePtr(),
                t2.Length(),
                output.GetDevicePtr(),
                output.Length()));
            cudaStreamSynchronize(0);
            return;
        }

        for (uint32_t n = 0; n < output.Batch(); ++n)
        {
            uint32_t t1N = min(n, t1.Batch() - 1);
            uint32_t t2N = min(n, t2.Batch() - 1);

            CUDA_CHECK(cublasSgeam(
                s_CublasHandle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                t1.BatchLength(),
                1,
                &alpha,
                t1.GetDevicePtr() + t1N * t1.BatchLength(),
                t1.BatchLength(),
                &beta,
                t2.GetDevicePtr() + t2N * t2.BatchLength(),
                t2.BatchLength(),
                output.GetDevicePtr() + n * output.BatchLength(),
                output.BatchLength()));
        }
        cudaStreamSynchronize(0);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::MatMul(bool transposeT1, bool transposeT2, const Tensor& t1, const Tensor& t2, Tensor& output) const
    {
        NVTXProfile nvtxProfile(__FUNCTION__, 0xFF004A7F);
        t1.CopyToDevice();
        t2.CopyToDevice();
        output.OverrideDevice();
        output.Zero();

        if (t1.Depth() == t2.Depth() && t1.Batch() == t2.Batch())
            MatMulStridedBatched(transposeT1, transposeT2, t1, t2, output);
        else if (t1.Depth() * output.Batch() > 48)
            MatMulBatched(transposeT1, transposeT2, t1, t2, output);
        else
            MatMulGeneric(transposeT1, transposeT2, t1, t2, output);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::Mul(float alpha, const Tensor& t1, float beta, const Tensor& t2, Tensor& output) const
    {
        NVTXProfile nvtxProfile(__FUNCTION__, 0xFF004A7F);
        t1.CopyToDevice();
        t2.CopyToDevice();
        output.OverrideDevice();

        if (t1.SameDimensionsOrOne(t2) || t2.SameDimensionsOrOne(t1))
        {
            cudnnOpTensorDescriptor_t mulDesc; cudnnCreateOpTensorDescriptor(&mulDesc);
            cudnnTensorDescriptor_t t1Desc; cudnnCreateTensorDescriptor(&t1Desc);
            cudnnTensorDescriptor_t t2Desc; cudnnCreateTensorDescriptor(&t2Desc);
            cudnnTensorDescriptor_t outputDesc; cudnnCreateTensorDescriptor(&outputDesc);

            cudnnSetOpTensorDescriptor(mulDesc, CUDNN_OP_TENSOR_MUL, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN);
            cudnnSetTensor4dDescriptor(t1Desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, t1.GetShape().Dimensions[3], t1.GetShape().Dimensions[2], t1.GetShape().Dimensions[1], t1.GetShape().Dimensions[0]);
            cudnnSetTensor4dDescriptor(t2Desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, t2.GetShape().Dimensions[3], t2.GetShape().Dimensions[2], t2.GetShape().Dimensions[1], t2.GetShape().Dimensions[0]);
            cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, output.GetShape().Dimensions[3], output.GetShape().Dimensions[2], output.GetShape().Dimensions[1], output.GetShape().Dimensions[0]);

            float beta2 = 0.f;
            if (t2.SameDimensionsOrOne(t1))
                CUDA_CHECK(cudnnOpTensor(s_CudnnHandle, mulDesc, &alpha, t1Desc, t1.GetDevicePtr(), &beta, t2Desc, t2.GetDevicePtr(), &beta2, outputDesc, output.GetDevicePtr()));
            else
                CUDA_CHECK(cudnnOpTensor(s_CudnnHandle, mulDesc, &beta, t2Desc, t2.GetDevicePtr(), &alpha, t1Desc, t1.GetDevicePtr(), &beta2, outputDesc, output.GetDevicePtr()));
            cudaStreamSynchronize(0);
            return;
        }

        dim3 blocks, threads;
        GetKernelRunParams(output.Length(), blocks, threads, s_CudaDevProp.maxThreadsPerBlock);

        if (t2.SameDimensionsExceptBatches(output))
        {
            CudaKernels::MulBroadcast(
                blocks,
                threads,
                beta,
                t2.GetDevicePtr(),
                alpha,
                t1.GetDevicePtr(),
                t1.Width(),
                t1.Height(),
                t1.Depth(),
                t1.Batch(),
                output.GetDevicePtr(),
                output.Width(),
                output.Height(),
                output.Depth(),
                output.Batch());
            cudaStreamSynchronize(0);
        }
        else if (t1.SameDimensionsExceptBatches(output))
        {
            CudaKernels::MulBroadcast(
                blocks,
                threads,
                alpha,
                t1.GetDevicePtr(),
                beta,
                t2.GetDevicePtr(),
                t2.Width(),
                t2.Height(),
                t2.Depth(),
                t2.Batch(),
                output.GetDevicePtr(),
                output.Width(),
                output.Height(),
                output.Depth(),
                output.Batch());
            cudaStreamSynchronize(0);
        }
        else
            __super::Mul(alpha, t1, beta, t2, output);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::Scale(Tensor& input, float v) const
    {
        NVTXProfile nvtxProfile(__FUNCTION__, 0xFF004A7F);
        input.CopyToDevice();

        cudnnTensorDescriptor_t inputDesc; cudnnCreateTensorDescriptor(&inputDesc);
        cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, input.GetShape().Dimensions[3], input.GetShape().Dimensions[2], input.GetShape().Dimensions[1], input.GetShape().Dimensions[0]);

        CUDA_CHECK(cudnnScaleTensor(s_CudnnHandle, inputDesc, input.GetDevicePtr(), &v));
        cudaStreamSynchronize(0);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::Div(float alpha, const Tensor& t1, float beta, const Tensor& t2, Tensor& output) const
    {
        NVTXProfile nvtxProfile(__FUNCTION__, 0xFF004A7F);
        t1.CopyToDevice();
        t2.CopyToDevice();
        output.OverrideDevice();

        dim3 blocks, threads;

        if (t1.GetShape() == t2.GetShape())
        {
            /*const int TMP = 32;
            GetKernelRunParams((int)ceil(t1.Length() / (float)TMP), blocks, threads, s_CudaDevProp.maxThreadsPerBlock);*/
            threads.x = 128;
            //blocks.x = min((t1.Length() + threads.x - 1) / threads.x, s_CudaDevProp.multiProcessorCount);
            blocks.x = (unsigned int)ceil((float)t1.Length() / threads.x / s_CudaDevProp.multiProcessorCount);
            CudaKernels::Div(blocks, threads, t1.Length(), alpha, t1.GetDevicePtr(), beta, t2.GetDevicePtr(), output.GetDevicePtr());
            cudaStreamSynchronize(0);
        }
        else
        {
            GetKernelRunParams(output.Length(), blocks, threads, s_CudaDevProp.maxThreadsPerBlock / 2);

            CudaKernels::DivBroadcast(
                blocks,
                threads,
                alpha,
                t1.GetDevicePtr(),
                t1.Width(),
                t1.Height(),
                t1.Depth(),
                t1.Batch(),
                beta,
                t2.GetDevicePtr(),
                t2.Width(),
                t2.Height(),
                t2.Depth(),
                t2.Batch(),
                output.GetDevicePtr(),
                output.Width(),
                output.Height(),
                output.Depth(),
                output.Batch());
            cudaStreamSynchronize(0);
        }
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::Mul(const Tensor& input, float v, Tensor& output) const
    {
        NVTXProfile nvtxProfile(__FUNCTION__, 0xFF004A7F);
        input.CopyToDevice();
        output.OverrideDevice();

        if (input.GetDevicePtr() == output.GetDevicePtr())
        {
            CUDA_CHECK(cublasSscal_v2(s_CublasHandle, input.Length(), &v, output.GetDevicePtr(), 1));
        }
        else
        {
            output.Zero();
            CUDA_CHECK(cublasSaxpy_v2(s_CublasHandle, input.Length(), &v, input.GetDevicePtr(), 1, output.GetDevicePtr(), 1));
        }
        cudaStreamSynchronize(0);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::Div(const Tensor& input, float v, Tensor& output) const
    {
        Mul(input, 1.f / v, output);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::Pow(const Tensor& input, float power, Tensor& output) const
    {
        NVTXProfile nvtxProfile(__FUNCTION__, 0xFF004A7F);
        input.CopyToDevice();
        output.OverrideDevice();

        if (power == 2)
        {
            cudnnOpTensorDescriptor_t sqrDesc; cudnnCreateOpTensorDescriptor(&sqrDesc);
            cudnnTensorDescriptor_t inputDesc; cudnnCreateTensorDescriptor(&inputDesc);
            cudnnTensorDescriptor_t outputDesc; cudnnCreateTensorDescriptor(&outputDesc);

            cudnnSetOpTensorDescriptor(sqrDesc, CUDNN_OP_TENSOR_MUL, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN);
            cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, input.GetShape().Dimensions[3], input.GetShape().Dimensions[2], input.GetShape().Dimensions[1], input.GetShape().Dimensions[0]);
            cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, output.GetShape().Dimensions[3], output.GetShape().Dimensions[2], output.GetShape().Dimensions[1], output.GetShape().Dimensions[0]);

            float alpha1 = 1.f, alpha2 = 1.f, beta = 0.f;
            CUDA_CHECK(cudnnOpTensor(s_CudnnHandle, sqrDesc, &alpha1, inputDesc, input.GetDevicePtr(), &alpha2, inputDesc, input.GetDevicePtr(), &beta, outputDesc, output.GetDevicePtr()));
            cudaStreamSynchronize(0);
            return;
        }

        dim3 blocks, threads;
        GetKernelRunParams((int)ceil(input.Length() / (float)INNER_KERNEL_LOOP_LENGTH), blocks, threads, s_CudaDevProp.maxThreadsPerBlock);        

        CudaKernels::Pow(blocks, threads, input.Length(), input.GetDevicePtr(), power, output.GetDevicePtr(), INNER_KERNEL_LOOP_LENGTH);
        cudaStreamSynchronize(0);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::PowGradient(const Tensor& input, float power, const Tensor& outputGradient, Tensor& inputGradient) const
    {
        NVTXProfile nvtxProfile(__FUNCTION__, 0xFF004A7F);
        input.CopyToDevice();
        outputGradient.CopyToDevice();
        inputGradient.OverrideDevice();

        if (power == 2)
        {
            Mul(2.f, outputGradient, 1.f, input, inputGradient);
            return;
        }

        dim3 blocks, threads;
        GetKernelRunParams((int)ceil(input.Length() / (float)INNER_KERNEL_LOOP_LENGTH), blocks, threads, s_CudaDevProp.maxThreadsPerBlock);

        CudaKernels::PowGradient(blocks, threads, input.Length(), input.GetDevicePtr(), power, outputGradient.GetDevicePtr(), inputGradient.GetDevicePtr(), INNER_KERNEL_LOOP_LENGTH);
        cudaStreamSynchronize(0);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::Sqrt(const Tensor& input, Tensor& output) const
    {
        NVTXProfile nvtxProfile(__FUNCTION__, 0xFF004A7F);
        input.CopyToDevice();
        output.OverrideDevice();

        cudnnOpTensorDescriptor_t sqrDesc; cudnnCreateOpTensorDescriptor(&sqrDesc);
        cudnnTensorDescriptor_t inputDesc; cudnnCreateTensorDescriptor(&inputDesc);
        cudnnTensorDescriptor_t outputDesc; cudnnCreateTensorDescriptor(&outputDesc);

        cudnnSetOpTensorDescriptor(sqrDesc, CUDNN_OP_TENSOR_SQRT, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN);
        cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, input.GetShape().Dimensions[3], input.GetShape().Dimensions[2], input.GetShape().Dimensions[1], input.GetShape().Dimensions[0]);
        cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, output.GetShape().Dimensions[3], output.GetShape().Dimensions[2], output.GetShape().Dimensions[1], output.GetShape().Dimensions[0]);

        float alpha1 = 1, alpha2 = 0, beta = 0;
        CUDA_CHECK(cudnnOpTensor(s_CudnnHandle, sqrDesc, &alpha1, inputDesc, input.GetDevicePtr(), &alpha2, inputDesc, input.GetDevicePtr(), &beta, outputDesc, output.GetDevicePtr()));
        cudaStreamSynchronize(0);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::Negate(const Tensor& input, Tensor& output) const
    {
        Mul(input, -1.f, output);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::Inverse(float alpha, const Tensor& input, Tensor& output) const
    {
        NVTXProfile nvtxProfile(__FUNCTION__, 0xFF004A7F);
        input.CopyToDevice();
        output.OverrideDevice();

        const int INV_SUB_LOOP_LEN = 32;

        dim3 blocks, threads;
        GetKernelRunParams((int)ceil(input.Length() / (float)INV_SUB_LOOP_LEN), blocks, threads, s_CudaDevProp.maxThreadsPerBlock);

        CudaKernels::Inverse(blocks, threads, input.Length(), input.GetDevicePtr(), alpha, output.GetDevicePtr(), INV_SUB_LOOP_LEN);
        cudaStreamSynchronize(0);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::Clip(const Tensor& input, float min, float max, Tensor& output) const
    {
        NVTXProfile nvtxProfile(__FUNCTION__, 0xFF004A7F);
        input.CopyToDevice();
        output.OverrideDevice();

        void* clipValPtr;
        DeviceMemoryManager::Default().Allocate(&clipValPtr, sizeof(float), "clip_value_tmp_tensor");

        cudnnOpTensorDescriptor_t minDesc; cudnnCreateOpTensorDescriptor(&minDesc);
        cudnnOpTensorDescriptor_t maxDesc; cudnnCreateOpTensorDescriptor(&maxDesc);
        cudnnTensorDescriptor_t inputDesc; cudnnCreateTensorDescriptor(&inputDesc);
        cudnnTensorDescriptor_t clipValDesc; cudnnCreateTensorDescriptor(&clipValDesc);
        cudnnTensorDescriptor_t outputDesc; cudnnCreateTensorDescriptor(&outputDesc);

        cudnnSetOpTensorDescriptor(minDesc, CUDNN_OP_TENSOR_MIN, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN);
        cudnnSetOpTensorDescriptor(maxDesc, CUDNN_OP_TENSOR_MAX, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN);
        cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, input.GetShape().Dimensions[3], input.GetShape().Dimensions[2], input.GetShape().Dimensions[1], input.GetShape().Dimensions[0]);
        cudnnSetTensor4dDescriptor(clipValDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 1, 1);
        cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, output.GetShape().Dimensions[3], output.GetShape().Dimensions[2], output.GetShape().Dimensions[1], output.GetShape().Dimensions[0]);

        float alpha = 1, beta = 1, beta2 = 0;
        // run min op with max value
        CUDA_CHECK(cudnnSetTensor(s_CudnnHandle, clipValDesc, clipValPtr, &max));
        CUDA_CHECK(cudnnOpTensor(s_CudnnHandle, minDesc, &alpha, inputDesc, input.GetDevicePtr(), &beta, clipValDesc, clipValPtr, &beta2, outputDesc, output.GetDevicePtr()));
        // run max op with min value
        CUDA_CHECK(cudnnSetTensor(s_CudnnHandle, clipValDesc, clipValPtr, &min));
        CUDA_CHECK(cudnnOpTensor(s_CudnnHandle, maxDesc, &alpha, outputDesc, output.GetDevicePtr(), &beta, clipValDesc, clipValPtr, &beta2, outputDesc, output.GetDevicePtr()));

        CUDA_CHECK(cudaLaunchHostFunc(0, DeallocateWorkspace, clipValPtr));
        cudaStreamSynchronize(0);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::Add(const Tensor& input, float v, Tensor& output) const
    {
        NVTXProfile nvtxProfile(__FUNCTION__, 0xFF004A7F);
        input.CopyToDevice();
        output.OverrideDevice();

        void* biasPtr;
        DeviceMemoryManager::Default().Allocate(&biasPtr, sizeof(float), "add_tmp_tensor");
        
        cudnnOpTensorDescriptor_t addDesc; cudnnCreateOpTensorDescriptor(&addDesc);
        cudnnTensorDescriptor_t inputDesc; cudnnCreateTensorDescriptor(&inputDesc);
        cudnnTensorDescriptor_t biasDesc; cudnnCreateTensorDescriptor(&biasDesc);
        cudnnTensorDescriptor_t outputDesc; cudnnCreateTensorDescriptor(&outputDesc);

        cudnnSetOpTensorDescriptor(addDesc, CUDNN_OP_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN);
        cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, input.GetShape().Dimensions[3], input.GetShape().Dimensions[2], input.GetShape().Dimensions[1], input.GetShape().Dimensions[0]);
        cudnnSetTensor4dDescriptor(biasDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 1, 1);
        cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, output.GetShape().Dimensions[3], output.GetShape().Dimensions[2], output.GetShape().Dimensions[1], output.GetShape().Dimensions[0]);

        float alpha = 1, beta = 1, beta2 = 0;
        CUDA_CHECK(cudnnSetTensor(s_CudnnHandle, biasDesc, biasPtr, &v));
        CUDA_CHECK(cudnnOpTensor(s_CudnnHandle, addDesc, &alpha, inputDesc, input.GetDevicePtr(), &beta, biasDesc, biasPtr, &beta2, outputDesc, output.GetDevicePtr()));

        CUDA_CHECK(cudaLaunchHostFunc(0, DeallocateWorkspace, biasPtr));
        cudaStreamSynchronize(0);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::Transpose(const Tensor& input, Tensor& output) const
    {
        NVTXProfile nvtxProfile(__FUNCTION__, 0xFF004A7F);
        input.CopyToDevice();
        output.OverrideDevice();

        int m = input.Height();
        uint32_t n = input.Width();

        //treat depth as batch
        int batches = input.Depth() * input.Batch();
        float alpha = 1, beta = 0;

        for (int b = 0; b < batches; ++b)
        {
            const float* tPtr = input.GetDevicePtr() + b * input.GetShape().Dim0Dim1;

            CUDA_CHECK(cublasSgeam(
                s_CublasHandle,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                m,
                n,  // trick to convert row major to column major
                &alpha,
                tPtr,
                n,
                &beta,
                tPtr,
                m,
                output.GetDevicePtr() + b * output.GetShape().Dim0Dim1, 
                m));
        }
        cudaStreamSynchronize(0);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::Conv2D(const Tensor& input, const Tensor& kernels, uint32_t stride, uint32_t paddingX, uint32_t paddingY, EDataFormat dataFormat, Tensor& output) const
    {
        NVTXProfile nvtxProfile(__FUNCTION__, 0xFF004A7F);
        input.CopyToDevice();
        kernels.CopyToDevice();
        output.OverrideDevice();

        cudnnConvolutionDescriptor_t convolutionDesc; cudnnCreateConvolutionDescriptor(&convolutionDesc);
        cudnnTensorDescriptor_t inputDesc; cudnnCreateTensorDescriptor(&inputDesc);
        cudnnFilterDescriptor_t kernelsDesc; cudnnCreateFilterDescriptor(&kernelsDesc);
        cudnnTensorDescriptor_t outputDesc; cudnnCreateTensorDescriptor(&outputDesc);

        cudnnSetConvolution2dDescriptor(convolutionDesc, paddingY, paddingX, stride, stride, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
        cudnnSetFilter4dDescriptor(kernelsDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, kernels.GetShape().Dimensions[3], kernels.GetShape().Dimensions[2], kernels.GetShape().Dimensions[1], kernels.GetShape().Dimensions[0]);
        if (dataFormat == NCHW)
        {
            cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, input.GetShape().Dimensions[3], input.GetShape().Dimensions[2], input.GetShape().Dimensions[1], input.GetShape().Dimensions[0]);
            cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, output.GetShape().Dimensions[3], output.GetShape().Dimensions[2], output.GetShape().Dimensions[1], output.GetShape().Dimensions[0]);
        }
        else
        {
            cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, input.GetShape().Dimensions[3], input.GetShape().Dimensions[0], input.GetShape().Dimensions[2], input.GetShape().Dimensions[1]);
            cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, output.GetShape().Dimensions[3], output.GetShape().Dimensions[0], output.GetShape().Dimensions[2], output.GetShape().Dimensions[1]);
        }

        cudnnConvolutionFwdAlgo_t algo;
        CUDA_CHECK(cudnnGetConvolutionForwardAlgorithm(s_CudnnHandle, inputDesc, kernelsDesc, convolutionDesc, outputDesc, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo));

        size_t workspaceSize;
        CUDA_CHECK(cudnnGetConvolutionForwardWorkspaceSize(s_CudnnHandle, inputDesc, kernelsDesc, convolutionDesc, outputDesc, algo, &workspaceSize));
        void* workspacePtr;
        DeviceMemoryManager::Default().Allocate(&workspacePtr, workspaceSize, "conv2d_workspace");

        float alpha = 1, beta = 0;
        CUDA_CHECK(cudnnConvolutionForward(
            s_CudnnHandle,
            &alpha,
            inputDesc,
            input.GetDevicePtr(),
            kernelsDesc,
            kernels.GetDevicePtr(),
            convolutionDesc,
            algo,
            workspacePtr,
            workspaceSize,
            &beta,
            outputDesc,
            output.GetDevicePtr()));

        CUDA_CHECK(cudaLaunchHostFunc(0, DeallocateWorkspace, workspacePtr));
        cudaStreamSynchronize(0);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::Conv2DBiasActivation(const Tensor& input, const Tensor& kernels, uint32_t stride, uint32_t paddingX, uint32_t paddingY, const Tensor& bias, EActivation activation, float activationAlpha, Tensor& output)
    {
        NVTXProfile nvtxProfile(__FUNCTION__, 0xFF004A7F);
        input.CopyToDevice();
        kernels.CopyToDevice();
        bias.CopyToDevice();
        output.OverrideDevice();

        cudnnConvolutionDescriptor_t convolutionDesc; cudnnCreateConvolutionDescriptor(&convolutionDesc);
        cudnnActivationDescriptor_t activationDesc; cudnnCreateActivationDescriptor(&activationDesc);
        cudnnTensorDescriptor_t inputDesc; cudnnCreateTensorDescriptor(&inputDesc);
        cudnnFilterDescriptor_t kernelsDesc; cudnnCreateFilterDescriptor(&kernelsDesc);
        cudnnTensorDescriptor_t biasDesc; cudnnCreateTensorDescriptor(&biasDesc);
        cudnnTensorDescriptor_t outputDesc; cudnnCreateTensorDescriptor(&outputDesc);

        cudnnSetConvolution2dDescriptor(convolutionDesc, paddingY, paddingX, stride, stride, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
        cudnnSetFilter4dDescriptor(kernelsDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, kernels.GetShape().Dimensions[3], kernels.GetShape().Dimensions[2], kernels.GetShape().Dimensions[1], kernels.GetShape().Dimensions[0]);
        cudnnSetTensor4dDescriptor(biasDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, bias.GetShape().Dimensions[3], bias.GetShape().Dimensions[2], bias.GetShape().Dimensions[1], bias.GetShape().Dimensions[0]);
        cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, input.GetShape().Dimensions[3], input.GetShape().Dimensions[2], input.GetShape().Dimensions[1], input.GetShape().Dimensions[0]);
        cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, output.GetShape().Dimensions[3], output.GetShape().Dimensions[2], output.GetShape().Dimensions[1], output.GetShape().Dimensions[0]);

        cudnnConvolutionFwdAlgo_t algo;
        CUDA_CHECK(cudnnGetConvolutionForwardAlgorithm(s_CudnnHandle, inputDesc, kernelsDesc, convolutionDesc, outputDesc, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo));

        cudnnSetActivationDescriptor(activationDesc, GetCudnnActivationMode(activation), CUDNN_NOT_PROPAGATE_NAN, activationAlpha);

        size_t workspaceSize;
        CUDA_CHECK(cudnnGetConvolutionForwardWorkspaceSize(s_CudnnHandle, inputDesc, kernelsDesc, convolutionDesc, outputDesc, algo, &workspaceSize));
        void* workspacePtr;
        DeviceMemoryManager::Default().Allocate(&workspacePtr, workspaceSize, "conv2d_workspace");

        float alpha = 1, beta = 0;
        CUDA_CHECK(cudnnConvolutionBiasActivationForward(
            s_CudnnHandle,
            &alpha,
            inputDesc,
            input.GetDevicePtr(),
            kernelsDesc,
            kernels.GetDevicePtr(),
            convolutionDesc,
            algo,
            workspacePtr,
            workspaceSize,
            &beta,
            outputDesc,
            output.GetDevicePtr(),
            biasDesc,
            bias.GetDevicePtr(),
            activationDesc,
            outputDesc,
            output.GetDevicePtr()));

        CUDA_CHECK(cudaLaunchHostFunc(0, DeallocateWorkspace, workspacePtr));
        cudaStreamSynchronize(0);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::Conv2DBiasGradient(const Tensor& gradient, Tensor& biasGradient)
    {
        NVTXProfile nvtxProfile(__FUNCTION__, 0xFF004A7F);
        gradient.CopyToDevice();
        biasGradient.OverrideDevice();
        biasGradient.Zero();

        cudnnTensorDescriptor_t gradientDesc; cudnnCreateTensorDescriptor(&gradientDesc);
        cudnnTensorDescriptor_t biasGradientDesc; cudnnCreateTensorDescriptor(&biasGradientDesc);

        cudnnSetTensor4dDescriptor(gradientDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, gradient.GetShape().Dimensions[3], gradient.GetShape().Dimensions[2], gradient.GetShape().Dimensions[1], gradient.GetShape().Dimensions[0]);
        cudnnSetTensor4dDescriptor(biasGradientDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, biasGradient.GetShape().Dimensions[3], biasGradient.GetShape().Dimensions[2], biasGradient.GetShape().Dimensions[1], biasGradient.GetShape().Dimensions[0]);

        float alpha = 1, beta = 0;
        CUDA_CHECK(cudnnConvolutionBackwardBias(
            s_CudnnHandle,
            &alpha,
            gradientDesc,
            gradient.GetDevicePtr(),
            &beta,
            biasGradientDesc,
            biasGradient.GetDevicePtr()));
        cudaStreamSynchronize(0);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::Conv2DInputGradient(const Tensor& gradient, const Tensor& kernels, uint32_t stride, uint32_t paddingX, uint32_t paddingY, EDataFormat dataFormat, Tensor& inputGradient) const
    {
        if (dataFormat == NHWC) // CuDNN doesn't support gradients for NHWC format
            return __super::Conv2DInputGradient(gradient, kernels, stride, paddingX, paddingY, dataFormat, inputGradient);

        NVTXProfile nvtxProfile(__FUNCTION__, 0xFF004A7F);
        gradient.CopyToDevice();
        kernels.CopyToDevice();
        inputGradient.OverrideDevice();
        inputGradient.Zero();

        cudnnConvolutionDescriptor_t convolutionDesc; cudnnCreateConvolutionDescriptor(&convolutionDesc);
        cudnnTensorDescriptor_t gradientDesc; cudnnCreateTensorDescriptor(&gradientDesc);
        cudnnFilterDescriptor_t kernelsDesc; cudnnCreateFilterDescriptor(&kernelsDesc);
        cudnnTensorDescriptor_t inputGradientDesc; cudnnCreateTensorDescriptor(&inputGradientDesc);

        cudnnSetConvolution2dDescriptor(convolutionDesc, paddingY, paddingX, stride, stride, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
        if (dataFormat == NCHW)
        {
            cudnnSetTensor4dDescriptor(gradientDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, gradient.GetShape().Dimensions[3], gradient.GetShape().Dimensions[2], gradient.GetShape().Dimensions[1], gradient.GetShape().Dimensions[0]);
            cudnnSetTensor4dDescriptor(inputGradientDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, inputGradient.GetShape().Dimensions[3], inputGradient.GetShape().Dimensions[2], inputGradient.GetShape().Dimensions[1], inputGradient.GetShape().Dimensions[0]);
        }
        else
        {
            cudnnSetTensor4dDescriptor(gradientDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, gradient.GetShape().Dimensions[3], gradient.GetShape().Dimensions[0], gradient.GetShape().Dimensions[2], gradient.GetShape().Dimensions[1]);
            cudnnSetTensor4dDescriptor(inputGradientDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, inputGradient.GetShape().Dimensions[3], inputGradient.GetShape().Dimensions[0], inputGradient.GetShape().Dimensions[2], inputGradient.GetShape().Dimensions[1]);
        }
        cudnnSetFilter4dDescriptor(kernelsDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, kernels.GetShape().Dimensions[3], kernels.GetShape().Dimensions[2], kernels.GetShape().Dimensions[1], kernels.GetShape().Dimensions[0]);

        cudnnConvolutionBwdDataAlgo_t algo;
        cudnnGetConvolutionBackwardDataAlgorithm(s_CudnnHandle, kernelsDesc, gradientDesc, convolutionDesc, inputGradientDesc, CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, &algo);

        size_t workspaceSize;
        cudnnGetConvolutionBackwardDataWorkspaceSize(s_CudnnHandle, kernelsDesc, gradientDesc, convolutionDesc, inputGradientDesc, algo, &workspaceSize);
        //inputGradient.m_GpuData.UpdateWorkspace(inputGradient.m_GpuData.m_ConvBackWorkspace, workspaceSize);
        void* workspacePtr;
        DeviceMemoryManager::Default().Allocate(&workspacePtr, workspaceSize, "conv2d_input_grad_workspace");

        float alpha = 1, beta = 0;
        CUDA_CHECK(cudnnConvolutionBackwardData(
            s_CudnnHandle,
            &alpha,
            kernelsDesc,
            kernels.GetDevicePtr(),
            gradientDesc,
            gradient.GetDevicePtr(),
            convolutionDesc,
            algo,
            workspacePtr,
            workspaceSize,
            &beta,
            inputGradientDesc,
            inputGradient.GetDevicePtr()));

        CUDA_CHECK(cudaLaunchHostFunc(0, DeallocateWorkspace, workspacePtr));
        cudaStreamSynchronize(0);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::Conv2DKernelsGradient(const Tensor& input, const Tensor& gradient, uint32_t stride, uint32_t paddingX, uint32_t paddingY, EDataFormat dataFormat, Tensor& kernelsGradient) const
    {
        if (dataFormat == NHWC) // CuDNN doesn't support gradients for NHWC format
            return __super::Conv2DKernelsGradient(input, gradient, stride, paddingX, paddingY, dataFormat, kernelsGradient);

        NVTXProfile nvtxProfile(__FUNCTION__, 0xFF004A7F);
        gradient.CopyToDevice();
        input.CopyToDevice();
        kernelsGradient.OverrideDevice();
        kernelsGradient.Zero();

        cudnnConvolutionDescriptor_t convolutionDesc; cudnnCreateConvolutionDescriptor(&convolutionDesc);
        cudnnTensorDescriptor_t gradientDesc; cudnnCreateTensorDescriptor(&gradientDesc);
        cudnnTensorDescriptor_t inputDesc; cudnnCreateTensorDescriptor(&inputDesc);
        cudnnFilterDescriptor_t kernelsGradientsDesc; cudnnCreateFilterDescriptor(&kernelsGradientsDesc);

        cudnnSetConvolution2dDescriptor(convolutionDesc, paddingY, paddingX, stride, stride, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);

        if (dataFormat == NCHW)
        {
            cudnnSetTensor4dDescriptor(gradientDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, gradient.GetShape().Dimensions[3], gradient.GetShape().Dimensions[2], gradient.GetShape().Dimensions[1], gradient.GetShape().Dimensions[0]);
            cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, input.GetShape().Dimensions[3], input.GetShape().Dimensions[2], input.GetShape().Dimensions[1], input.GetShape().Dimensions[0]);
        }
        else
        {
            cudnnSetTensor4dDescriptor(gradientDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, gradient.GetShape().Dimensions[3], gradient.GetShape().Dimensions[0], gradient.GetShape().Dimensions[2], gradient.GetShape().Dimensions[1]);
            cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, input.GetShape().Dimensions[3], input.GetShape().Dimensions[0], input.GetShape().Dimensions[2], input.GetShape().Dimensions[1]);
        }
        cudnnSetFilter4dDescriptor(kernelsGradientsDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, kernelsGradient.GetShape().Dimensions[3], kernelsGradient.GetShape().Dimensions[2], kernelsGradient.GetShape().Dimensions[1], kernelsGradient.GetShape().Dimensions[0]);

        cudnnConvolutionBwdFilterAlgo_t algo;
        cudnnGetConvolutionBackwardFilterAlgorithm(s_CudnnHandle, inputDesc, gradientDesc, convolutionDesc, kernelsGradientsDesc, CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &algo);

        size_t workspaceSize;
        cudnnGetConvolutionBackwardFilterWorkspaceSize(s_CudnnHandle, inputDesc, gradientDesc, convolutionDesc, kernelsGradientsDesc, algo, &workspaceSize);
        void* workspacePtr;
        DeviceMemoryManager::Default().Allocate(&workspacePtr, workspaceSize, "conv2d_kernels_grad_workspace");

        float alpha = 1, beta = 0;
        CUDA_CHECK(cudnnConvolutionBackwardFilter(
            s_CudnnHandle,
            &alpha,
            inputDesc,
            input.GetDevicePtr(),
            gradientDesc,
            gradient.GetDevicePtr(),
            convolutionDesc,
            algo,
            workspacePtr,
            workspaceSize,
            &beta,
            kernelsGradientsDesc,
            kernelsGradient.GetDevicePtr()));

        CUDA_CHECK(cudaLaunchHostFunc(0, DeallocateWorkspace, workspacePtr));
        cudaStreamSynchronize(0);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::Pool2D(const Tensor& input, uint32_t filterSize, uint32_t stride, EPoolingMode type, uint32_t paddingX, uint32_t paddingY, EDataFormat dataFormat, Tensor& output) const
    {
        NVTXProfile nvtxProfile(__FUNCTION__, 0xFF004A7F);
        input.CopyToDevice();
        output.OverrideDevice();

        cudnnPoolingDescriptor_t poolingDesc; cudnnCreatePoolingDescriptor(&poolingDesc);
        cudnnTensorDescriptor_t inputDesc; cudnnCreateTensorDescriptor(&inputDesc);
        cudnnTensorDescriptor_t outputDesc; cudnnCreateTensorDescriptor(&outputDesc);

        cudnnSetPooling2dDescriptor(poolingDesc, GetCudnnPoolType(type), CUDNN_NOT_PROPAGATE_NAN, filterSize, filterSize, paddingX, paddingY, stride, stride);
        if (dataFormat == NCHW)
        {
            cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, input.GetShape().Dimensions[3], input.GetShape().Dimensions[2], input.GetShape().Dimensions[1], input.GetShape().Dimensions[0]);
            cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, output.GetShape().Dimensions[3], output.GetShape().Dimensions[2], output.GetShape().Dimensions[1], output.GetShape().Dimensions[0]);
        }
        else
        {
            cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, input.GetShape().Dimensions[3], input.GetShape().Dimensions[0], input.GetShape().Dimensions[2], input.GetShape().Dimensions[1]);
            cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, output.GetShape().Dimensions[3], output.GetShape().Dimensions[0], output.GetShape().Dimensions[2], output.GetShape().Dimensions[1]);
        }

        float alpha = 1, beta = 0;
        CUDA_CHECK(cudnnPoolingForward(
            s_CudnnHandle,
            poolingDesc,
            &alpha,
            inputDesc,
            input.GetDevicePtr(),
            &beta,
            outputDesc,
            output.GetDevicePtr()));
        cudaStreamSynchronize(0);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::Pool2DGradient(const Tensor& output, const Tensor& input, const Tensor& outputGradient, uint32_t filterSize, uint32_t stride, EPoolingMode type, uint32_t paddingX, uint32_t paddingY, EDataFormat dataFormat, Tensor& inputGradient) const
    {
        NVTXProfile nvtxProfile(__FUNCTION__, 0xFF004A7F);
        output.CopyToDevice();
        input.CopyToDevice();
        outputGradient.CopyToDevice();
        inputGradient.OverrideDevice();
        inputGradient.Zero();

        cudnnPoolingDescriptor_t poolingDesc; cudnnCreatePoolingDescriptor(&poolingDesc);
        cudnnTensorDescriptor_t outputDesc; cudnnCreateTensorDescriptor(&outputDesc);
        cudnnTensorDescriptor_t inputDesc; cudnnCreateTensorDescriptor(&inputDesc);
        cudnnTensorDescriptor_t outputGradientDesc; cudnnCreateTensorDescriptor(&outputGradientDesc);
        cudnnTensorDescriptor_t inputGradientDesc; cudnnCreateTensorDescriptor(&inputGradientDesc);

        cudnnSetPooling2dDescriptor(poolingDesc, GetCudnnPoolType(type), CUDNN_NOT_PROPAGATE_NAN, filterSize, filterSize, paddingX, paddingY, stride, stride);
        if (dataFormat == NCHW)
        {
            cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, output.GetShape().Dimensions[3], output.GetShape().Dimensions[2], output.GetShape().Dimensions[1], output.GetShape().Dimensions[0]);
            cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, input.GetShape().Dimensions[3], input.GetShape().Dimensions[2], input.GetShape().Dimensions[1], input.GetShape().Dimensions[0]);
            cudnnSetTensor4dDescriptor(outputGradientDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, outputGradient.GetShape().Dimensions[3], outputGradient.GetShape().Dimensions[2], outputGradient.GetShape().Dimensions[1], outputGradient.GetShape().Dimensions[0]);
            cudnnSetTensor4dDescriptor(inputGradientDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, inputGradient.GetShape().Dimensions[3], inputGradient.GetShape().Dimensions[2], inputGradient.GetShape().Dimensions[1], inputGradient.GetShape().Dimensions[0]);
        }
        else
        {
            cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, output.GetShape().Dimensions[3], output.GetShape().Dimensions[0], output.GetShape().Dimensions[2], output.GetShape().Dimensions[1]);
            cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, input.GetShape().Dimensions[3], input.GetShape().Dimensions[0], input.GetShape().Dimensions[2], input.GetShape().Dimensions[1]);
            cudnnSetTensor4dDescriptor(outputGradientDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, outputGradient.GetShape().Dimensions[3], outputGradient.GetShape().Dimensions[0], outputGradient.GetShape().Dimensions[2], outputGradient.GetShape().Dimensions[1]);
            cudnnSetTensor4dDescriptor(inputGradientDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, inputGradient.GetShape().Dimensions[3], inputGradient.GetShape().Dimensions[0], inputGradient.GetShape().Dimensions[2], inputGradient.GetShape().Dimensions[1]);
        }

        float alpha = 1, beta = 0;
        CUDA_CHECK(cudnnPoolingBackward(
            s_CudnnHandle,
            poolingDesc,
            &alpha,
            outputDesc,
            output.GetDevicePtr(),
            outputGradientDesc,
            outputGradient.GetDevicePtr(),
            inputDesc,
            input.GetDevicePtr(),
            &beta,
            inputGradientDesc,
            inputGradient.GetDevicePtr()));
        cudaStreamSynchronize(0);
    }

    ////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::UpSample2D(const Tensor& input, uint32_t scaleFactor, Tensor& output) const
    {
        Tensor tmp(output.GetShape());
        tmp.TryDeviceAllocate();
        tmp.OverrideDevice();
        Pool2DGradient(tmp, tmp, input, scaleFactor, scaleFactor, AvgPool, 0, 0, NCHW, output);
        Scale(output, (float)scaleFactor * scaleFactor);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::UpSample2DGradient(const Tensor& outputGradient, uint32_t scaleFactor, Tensor& inputGradient) const
    {
        Pool2D(outputGradient, scaleFactor, scaleFactor, AvgPool, 0, 0, NCHW, inputGradient);
        Scale(inputGradient, (float)scaleFactor * scaleFactor);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::BatchNormalization(const Tensor& input, EBatchNormMode mode, const Tensor& gamma, const Tensor& beta, float epsilon, const Tensor* runningMean, const Tensor* runningVar, Tensor& output) const
    {
        if (mode == Instance)
            return __super::BatchNormalization(input, mode, gamma, beta, epsilon, runningMean, runningVar, output);

        NVTXProfile nvtxProfile(__FUNCTION__, 0xFF004A7F);
        input.CopyToDevice();
        gamma.CopyToDevice();
        beta.CopyToDevice();
        if (runningMean)
            runningMean->CopyToDevice();
        if (runningVar)
            runningVar->CopyToDevice();
        output.OverrideDevice();

        cudnnTensorDescriptor_t inputOutputDesc; cudnnCreateTensorDescriptor(&inputOutputDesc);
        cudnnTensorDescriptor_t gammaBetaMeanVarDesc; cudnnCreateTensorDescriptor(&gammaBetaMeanVarDesc);

        cudnnSetTensor4dDescriptor(inputOutputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, input.GetShape().Dimensions[3], input.GetShape().Dimensions[2], input.GetShape().Dimensions[1], input.GetShape().Dimensions[0]);
        cudnnSetTensor4dDescriptor(gammaBetaMeanVarDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, gamma.GetShape().Dimensions[3], gamma.GetShape().Dimensions[2], gamma.GetShape().Dimensions[1], gamma.GetShape().Dimensions[0]);

        float alpha = 1, _beta = 0;
        CUDA_CHECK(cudnnBatchNormalizationForwardInference(
            s_CudnnHandle,
            GetCudnnBatchNormMode(mode),
            &alpha,
            &_beta,
            inputOutputDesc,
            input.GetDevicePtr(),
            inputOutputDesc,
            output.GetDevicePtr(),
            gammaBetaMeanVarDesc,
            gamma.GetDevicePtr(),
            beta.GetDevicePtr(),
            runningMean ? runningMean->GetDevicePtr() : nullptr,
            runningVar ? runningVar->GetDevicePtr() : nullptr,
            epsilon));
        cudaStreamSynchronize(0);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::BatchNormalizationTrain(const Tensor& input, EBatchNormMode mode, const Tensor& gamma, const Tensor& beta, float momentum, float epsilon, Tensor* runningMean, Tensor* runningVar, Tensor& saveMean, Tensor& saveInvVariance, Tensor& output) const
    {
        if (mode == Instance)
            return __super::BatchNormalizationTrain(input, mode, gamma, beta, momentum, epsilon, runningMean, runningVar, saveMean, saveInvVariance, output);

        NVTXProfile nvtxProfile(__FUNCTION__, 0xFF004A7F);
        input.CopyToDevice();
        gamma.CopyToDevice();
        beta.CopyToDevice();
        if (runningMean)
            runningMean->CopyToDevice();
        if (runningVar)
            runningVar->CopyToDevice();
        saveMean.OverrideDevice();
        saveInvVariance.OverrideDevice();
        output.OverrideDevice();

        cudnnTensorDescriptor_t inputOutputDesc; cudnnCreateTensorDescriptor(&inputOutputDesc);
        cudnnTensorDescriptor_t gammaBetaMeanVarDesc; cudnnCreateTensorDescriptor(&gammaBetaMeanVarDesc);

        cudnnSetTensor4dDescriptor(inputOutputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, input.GetShape().Dimensions[3], input.GetShape().Dimensions[2], input.GetShape().Dimensions[1], input.GetShape().Dimensions[0]);
        cudnnSetTensor4dDescriptor(gammaBetaMeanVarDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, gamma.GetShape().Dimensions[3], gamma.GetShape().Dimensions[2], gamma.GetShape().Dimensions[1], gamma.GetShape().Dimensions[0]);

        float alpha = 1, _beta = 0;
        CUDA_CHECK(cudnnBatchNormalizationForwardTraining(
            s_CudnnHandle,
            GetCudnnBatchNormMode(mode),
            &alpha,
            &_beta,
            inputOutputDesc,
            input.GetDevicePtr(),
            inputOutputDesc,
            output.GetDevicePtr(),
            gammaBetaMeanVarDesc,
            gamma.GetDevicePtr(),
            beta.GetDevicePtr(),
            momentum,
            runningMean ? runningMean->GetDevicePtr() : nullptr,
            runningVar ? runningVar->GetDevicePtr() : nullptr,
            epsilon,
            saveMean.GetDevicePtr(),
            saveInvVariance.GetDevicePtr()));
        cudaStreamSynchronize(0);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::BatchNormalizationGradient(const Tensor& input, EBatchNormMode mode, const Tensor& gamma, float epsilon, const Tensor& outputGradient, const Tensor& savedMean, const Tensor& savedInvVariance, Tensor& gammaGradient, Tensor& betaGradient, bool trainable, Tensor& inputGradient) const
    {
        if (mode == Instance)
            return __super::BatchNormalizationGradient(input, mode, gamma, epsilon, outputGradient, savedMean, savedInvVariance, gammaGradient, betaGradient, trainable, inputGradient);

        NVTXProfile nvtxProfile(__FUNCTION__, 0xFF004A7F);
        input.CopyToDevice();
        gamma.CopyToDevice();
        outputGradient.CopyToDevice();
        savedMean.CopyToDevice();
        savedInvVariance.CopyToDevice();
        gammaGradient.OverrideDevice();
        gammaGradient.Zero();
        betaGradient.OverrideDevice();
        betaGradient.Zero();
        inputGradient.OverrideDevice();
        inputGradient.Zero();

        cudnnTensorDescriptor_t inputOutputGradientDesc; cudnnCreateTensorDescriptor(&inputOutputGradientDesc);
        cudnnTensorDescriptor_t gammaBetaGradientDesc; cudnnCreateTensorDescriptor(&gammaBetaGradientDesc);

        cudnnSetTensor4dDescriptor(inputOutputGradientDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, input.GetShape().Dimensions[3], input.GetShape().Dimensions[2], input.GetShape().Dimensions[1], input.GetShape().Dimensions[0]);
        cudnnSetTensor4dDescriptor(gammaBetaGradientDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, gamma.GetShape().Dimensions[3], gamma.GetShape().Dimensions[2], gamma.GetShape().Dimensions[1], gamma.GetShape().Dimensions[0]);

        float alpha = 1.f, beta = 0.f, paramsGradAlpha = (trainable ? 1.f : 0.f), paramsGradBeta = 0.f;
        CUDA_CHECK(cudnnBatchNormalizationBackward(
            s_CudnnHandle,
            GetCudnnBatchNormMode(mode),
            &alpha,
            &beta,
            &paramsGradAlpha,
            &paramsGradBeta,
            inputOutputGradientDesc,
            input.GetDevicePtr(),
            inputOutputGradientDesc,
            outputGradient.GetDevicePtr(),
            inputOutputGradientDesc,
            inputGradient.GetDevicePtr(),
            gammaBetaGradientDesc,
            gamma.GetDevicePtr(),
            gammaGradient.GetDevicePtr(),
            betaGradient.GetDevicePtr(),
            epsilon,
            savedMean.GetDevicePtr(),
            savedInvVariance.GetDevicePtr()));
        cudaStreamSynchronize(0);
    }

    //////////////////////////////////////////////////////////////////////////
    //void TensorOpGpu::Dropout(const Tensor& input, float prob, Tensor& saveMask, void** states, Tensor& output)
    //{
    //    NVTXProfile nvtxProfile(__FUNCTION__, 0xFF004A7F);
    //    input.CopyToDevice();
    //    output.OverrideDevice();
    //    prob = 1 - prob;

    //    cudnnTensorDescriptor_t inputOutputDesc; cudnnCreateTensorDescriptor(&inputOutputDesc);
    //    cudnnDropoutDescriptor_t dropoutDesc; cudnnCreateDropoutDescriptor(&dropoutDesc);

    //    cudnnSetTensor4dDescriptor(inputOutputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, input.GetShape().Dimensions[3], input.GetShape().Dimensions[2], input.GetShape().Dimensions[1], input.GetShape().Dimensions[0]);

    //    size_t dropoutStateSize;
    //    CUDA_CHECK(cudnnDropoutGetStatesSize(s_CudnnHandle, &dropoutStateSize));
    //    if (!states)
    //    MemoryManager::Default().Allocate(states, dropoutStateSize);

    //    size_t dropoutReserveSize;
    //    CUDA_CHECK(cudnnDropoutGetReserveSpaceSize(inputOutputDesc, &dropoutReserveSize));
    //    saveMask.m_GpuData.UpdateWorkspace(saveMask.m_GpuData.m_DropoutWorkspace, dropoutReserveSize);

    //    saveMask(0) = prob;

    //    cudnnSetDropoutDescriptor(dropoutDesc, s_CudnnHandle, prob, *states, dropoutStateSize, 0);

    //    CUDA_CHECK(cudnnDropoutForward(
    //        s_CudnnHandle,
    //        dropoutDesc,
    //        inputOutputDesc,
    //        input.GetDevicePtr(),
    //        inputOutputDesc,
    //        output.GetDevicePtr(),
    //        saveMask.m_GpuData.m_DropoutWorkspace->GetDevicePtr(),
    //        dropoutReserveSize));
    //    cudaStreamSynchronize(0);
    //}

    //////////////////////////////////////////////////////////////////////////////
    //void TensorOpGpu::DropoutGradient(const Tensor& outputGradient, const Tensor& savedMask, Tensor& inputGradient)
    //{
    //    NVTXProfile nvtxProfile(__FUNCTION__, 0xFF004A7F);
    //    outputGradient.CopyToDevice();
    //    inputGradient.OverrideDevice();
    //    inputGradient.Zero();

    //    cudnnTensorDescriptor_t inputOutputGradDesc; cudnnCreateTensorDescriptor(&inputOutputGradDesc);
    //    cudnnDropoutDescriptor_t dropoutDesc; cudnnCreateDropoutDescriptor(&dropoutDesc);

    //    cudnnSetTensor4dDescriptor(inputOutputGradDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, outputGradient.GetShape().Dimensions[3], outputGradient.GetShape().Dimensions[2], outputGradient.GetShape().Dimensions[1], outputGradient.GetShape().Dimensions[0]);
    //    
    //    size_t dropoutStateSize;
    //    CUDA_CHECK(cudnnDropoutGetStatesSize(s_CudnnHandle, &dropoutStateSize));
    //    size_t dropoutReserveSize;
    //    CUDA_CHECK(cudnnDropoutGetReserveSpaceSize(inputOutputGradDesc, &dropoutReserveSize));

    //    cudnnSetDropoutDescriptor(dropoutDesc, s_CudnnHandle, savedMask(0), states, dropoutStateSize, 0);

    //    CUDA_CHECK(cudnnDropoutBackward(
    //        s_CudnnHandle,
    //        dropoutDesc,
    //        inputOutputGradDesc,
    //        outputGradient.GetDevicePtr(),
    //        inputOutputGradDesc,
    //        inputGradient.GetDevicePtr(),
    //        savedMask.m_GpuData.m_DropoutWorkspace->GetDevicePtr(),
    //        dropoutReserveSize));
    //    cudaStreamSynchronize(0);
    //}

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::Sigmoid(const Tensor& input, Tensor& output) const
    {
        Activation(CUDNN_ACTIVATION_SIGMOID, input, output, 0);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::SigmoidGradient(const Tensor& output, const Tensor& outputGradient, Tensor& inputGradient) const
    {
        ActivationGradient(CUDNN_ACTIVATION_SIGMOID, output, outputGradient, inputGradient, 0);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::Tanh(const Tensor& input, Tensor& output) const
    {
        Activation(CUDNN_ACTIVATION_TANH, input, output, 0);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::TanhGradient(const Tensor& output, const Tensor& outputGradient, Tensor& inputGradient) const
    {
        ActivationGradient(CUDNN_ACTIVATION_TANH, output, outputGradient, inputGradient, 0);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::ReLU(const Tensor& input, Tensor& output) const
    {
        Activation(CUDNN_ACTIVATION_RELU, input, output, 0);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::ReLUGradient(const Tensor& output, const Tensor& outputGradient, Tensor& inputGradient) const
    {
        ActivationGradient(CUDNN_ACTIVATION_RELU, output, outputGradient, inputGradient, 0);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::Elu(const Tensor& input, float alpha, Tensor& output) const
    {
        Activation(CUDNN_ACTIVATION_ELU, input, output, alpha);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::EluGradient(const Tensor& output, const Tensor& outputGradient, float alpha, Tensor& inputGradient) const
    {
        ActivationGradient(CUDNN_ACTIVATION_ELU, output, outputGradient, inputGradient, alpha);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::LeakyReLU(const Tensor& input, float alpha, Tensor& output) const
    {
        NVTXProfile nvtxProfile(__FUNCTION__, 0xFF004A7F);
        dim3 blocks, threads;
        GetKernelRunParams(input.Length(), blocks, threads, s_CudaDevProp.maxThreadsPerBlock);
        input.CopyToDevice();
        output.OverrideDevice();

        CudaKernels::LeakyReLU(blocks, threads, input.Length(), input.GetDevicePtr(), alpha, output.GetDevicePtr());
        cudaStreamSynchronize(0);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::LeakyReLUGradient(const Tensor& output, const Tensor& outputGradient, float alpha, Tensor& inputGradient) const
    {
        NVTXProfile nvtxProfile(__FUNCTION__, 0xFF004A7F);
        dim3 blocks, threads;
        GetKernelRunParams(output.Length(), blocks, threads, s_CudaDevProp.maxThreadsPerBlock);
        output.CopyToDevice();
        outputGradient.CopyToDevice();
        inputGradient.OverrideDevice();
        inputGradient.Zero();

        CudaKernels::LeakyReLUGradient(blocks, threads, output.Length(), output.GetDevicePtr(), outputGradient.GetDevicePtr(), alpha, inputGradient.GetDevicePtr());
        cudaStreamSynchronize(0);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::Softmax(const Tensor& input, Tensor& output) const
    {
        NVTXProfile nvtxProfile(__FUNCTION__, 0xFF004A7F);
        input.CopyToDevice();
        output.OverrideDevice();

        cudnnTensorDescriptor_t inputDesc; cudnnCreateTensorDescriptor(&inputDesc);
        cudnnTensorDescriptor_t outputDesc; cudnnCreateTensorDescriptor(&outputDesc);

        uint32_t n = input.Batch(), c = input.Depth(), h = input.Height(), w = input.Width();

        cudnnSetTensor4dDescriptor(inputDesc,CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);
        cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);

        float alpha = 1, beta = 0;
        CUDA_CHECK(cudnnSoftmaxForward(
            s_CudnnHandle,
            CUDNN_SOFTMAX_ACCURATE,
            CUDNN_SOFTMAX_MODE_INSTANCE,
            &alpha,
            inputDesc,
            input.GetDevicePtr(),
            &beta,
            outputDesc,
            output.GetDevicePtr()));
        cudaStreamSynchronize(0);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::SoftmaxGradient(const Tensor& output, const Tensor& outputGradient, Tensor& inputGradient) const
    {
        NVTXProfile nvtxProfile(__FUNCTION__, 0xFF004A7F);
        output.CopyToDevice();
        outputGradient.CopyToDevice();
        inputGradient.OverrideDevice();
        inputGradient.Zero();

        cudnnTensorDescriptor_t outputDesc; cudnnCreateTensorDescriptor(&outputDesc);
        cudnnTensorDescriptor_t outputGradientDesc; cudnnCreateTensorDescriptor(&outputGradientDesc);
        cudnnTensorDescriptor_t inputGradientDesc; cudnnCreateTensorDescriptor(&inputGradientDesc);

        uint32_t n = output.Batch(), c = output.Depth(), h = output.Height(), w = output.Width();

        cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);
        cudnnSetTensor4dDescriptor(outputGradientDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);
        cudnnSetTensor4dDescriptor(inputGradientDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);

        float alpha = 1, beta = 0;
        CUDA_CHECK(cudnnSoftmaxBackward(
            s_CudnnHandle,
            CUDNN_SOFTMAX_ACCURATE,
            CUDNN_SOFTMAX_MODE_INSTANCE,
            &alpha,
            outputDesc,
            output.GetDevicePtr(),
            outputGradientDesc,
            outputGradient.GetDevicePtr(),
            &beta,
            inputGradientDesc,
            inputGradient.GetDevicePtr()));
        cudaStreamSynchronize(0);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::Reduce(const Tensor& input, cudnnReduceTensorOp_t reductionOp, Tensor& output) const
    {
        NVTXProfile nvtxProfile(__FUNCTION__, 0xFF004A7F);
        input.CopyToDevice();
        output.OverrideDevice();

        cudnnReduceTensorDescriptor_t reduceDesc; cudnnCreateReduceTensorDescriptor(&reduceDesc);
        cudnnTensorDescriptor_t inputDesc; cudnnCreateTensorDescriptor(&inputDesc);
        cudnnTensorDescriptor_t outputDesc; cudnnCreateTensorDescriptor(&outputDesc);

        cudnnSetReduceTensorDescriptor(reduceDesc, reductionOp, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN, CUDNN_REDUCE_TENSOR_NO_INDICES, CUDNN_32BIT_INDICES);
        cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, input.GetShape().Dimensions[3], input.GetShape().Dimensions[2], input.GetShape().Dimensions[1], input.GetShape().Dimensions[0]);
        cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, output.GetShape().Dimensions[3], output.GetShape().Dimensions[2], output.GetShape().Dimensions[1], output.GetShape().Dimensions[0]);

        size_t workspaceSize;
        CUDA_CHECK(cudnnGetReductionWorkspaceSize(s_CudnnHandle, reduceDesc, inputDesc, outputDesc, &workspaceSize));
        void* workspacePtr;
        DeviceMemoryManager::Default().Allocate(&workspacePtr, workspaceSize, "reduce_workspace");

        float alpha = 1, beta = 0;
        CUDA_CHECK(cudnnReduceTensor(
            s_CudnnHandle,
            reduceDesc,
            nullptr,
            0,
            workspacePtr,
            workspaceSize,
            &alpha,
            inputDesc,
            input.GetDevicePtr(),
            &beta,
            outputDesc,
            output.GetDevicePtr()));

        CUDA_CHECK(cudaLaunchHostFunc(0, DeallocateWorkspace, workspacePtr));
        cudaStreamSynchronize(0);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::AbsSum(const Tensor& input, EAxis axis, Tensor& output) const
    {
        Reduce(input, CUDNN_REDUCE_TENSOR_NORM1, output);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::Sum(const Tensor& input, EAxis axis, Tensor& output) const
    {
        Reduce(input, CUDNN_REDUCE_TENSOR_ADD, output);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::Mean(const Tensor& input, EAxis axis, Tensor& output) const
    {
        Reduce(input, CUDNN_REDUCE_TENSOR_AVG, output);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::AdamStep(Tensor& parameter, const Tensor& gradient, Tensor& mGrad, Tensor& vGrad, /*float batchSize, */float lr, float beta1, float beta2, float epsilon) const
    {
        NVTXProfile nvtxProfile(__FUNCTION__, 0xFF004A7F);
        parameter.CopyToDevice();
        gradient.CopyToDevice();
        mGrad.CopyToDevice();
        vGrad.CopyToDevice();

        Add(beta1, mGrad, 1 - beta1, gradient, mGrad);

        Tensor tmp(gradient.GetShape());
        tmp.OverrideDevice();
        Pow(gradient, 2, tmp);

        Add(beta2, vGrad, 1 - beta2, tmp, vGrad);

        Sqrt(vGrad, tmp);
        Add(tmp, epsilon, tmp);
        Div(1.f, mGrad, 1.f, tmp, tmp);
        Scale(tmp, lr);
        Sub(parameter, tmp, parameter);
        
        //dim3 blocks, threads;
        //GetKernelRunParams(parameter.Length(), blocks, threads, s_CudaDevProp.maxThreadsPerBlock);
        //
        //CudaKernels::AdamStep(blocks, threads, parameter.Length(), parameter.GetDevicePtr(), gradient.GetDevicePtr(), mGrad.GetDevicePtr(), vGrad.GetDevicePtr(), /*batchSize, */lr, beta1, beta2, epsilon);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::SgdStep(Tensor& parameter, const Tensor& gradient, /*float batchSize, */float lr) const
    {
        NVTXProfile nvtxProfile(__FUNCTION__, 0xFF004A7F);
        parameter.CopyToDevice();
        gradient.CopyToDevice();

        dim3 blocks, threads;
        GetKernelRunParams(parameter.Length(), blocks, threads, s_CudaDevProp.maxThreadsPerBlock);

        CudaKernels::SgdStep(blocks, threads, parameter.Length(), parameter.GetDevicePtr(), gradient.GetDevicePtr(), /*batchSize, */lr);
        cudaStreamSynchronize(0);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::Activation(const cudnnActivationMode_t& activationMode, const Tensor& input, Tensor& output, float coeff) const
    {
        NVTXProfile nvtxProfile(__FUNCTION__, 0xFF004A7F);
        input.CopyToDevice();
        output.OverrideDevice();

        cudnnTensorDescriptor_t inputDesc; cudnnCreateTensorDescriptor(&inputDesc);
        cudnnTensorDescriptor_t outputDesc; cudnnCreateTensorDescriptor(&outputDesc);
        cudnnActivationDescriptor_t activationDesc; cudnnCreateActivationDescriptor(&activationDesc);

        uint32_t n = input.Batch(), c = input.Depth(), h = input.Height(), w = input.Width();

        cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);
        cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);
        cudnnSetActivationDescriptor(activationDesc, activationMode, CUDNN_PROPAGATE_NAN, coeff);

        float alpha = 1, beta = 0;
        CUDA_CHECK(cudnnActivationForward(
            s_CudnnHandle,
            activationDesc,
            &alpha,
            inputDesc,
            input.GetDevicePtr(),
            &beta,
            outputDesc,
            output.GetDevicePtr()));
        cudaStreamSynchronize(0);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::ActivationGradient(const cudnnActivationMode_t& activationMode, const Tensor& output, const Tensor& outputGradient, Tensor& inputGradient, float coeff) const
    {
        NVTXProfile nvtxProfile(__FUNCTION__, 0xFF004A7F);
        output.CopyToDevice();
        outputGradient.CopyToDevice();
        inputGradient.OverrideDevice();
        inputGradient.Zero();

        cudnnTensorDescriptor_t outputDesc; cudnnCreateTensorDescriptor(&outputDesc);
        cudnnTensorDescriptor_t outputGradientDesc; cudnnCreateTensorDescriptor(&outputGradientDesc);
        cudnnTensorDescriptor_t inputGradientDesc; cudnnCreateTensorDescriptor(&inputGradientDesc);
        cudnnActivationDescriptor_t activationDesc; cudnnCreateActivationDescriptor(&activationDesc);

        uint32_t n = output.Batch(), c = output.Depth(), h = output.Height(), w = output.Width();

        cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);
        cudnnSetTensor4dDescriptor(outputGradientDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);
        cudnnSetTensor4dDescriptor(inputGradientDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);
        cudnnSetActivationDescriptor(activationDesc, activationMode, CUDNN_PROPAGATE_NAN, coeff);

        float alpha = 1, beta = 0;
        CUDA_CHECK(cudnnActivationBackward(
            s_CudnnHandle,
            activationDesc,
            &alpha,
            outputDesc,
            output.GetDevicePtr(),
            outputGradientDesc,
            outputGradient.GetDevicePtr(),
            outputDesc,
            output.GetDevicePtr(),
            &beta,
            inputGradientDesc,
            inputGradient.GetDevicePtr()));
        cudaStreamSynchronize(0);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::MatMulGeneric(bool transposeT1, bool transposeT2, const Tensor& t1, const Tensor& t2, Tensor& output) const
    {
        NVTXProfile nvtxProfile(__FUNCTION__, 0xFF004A7F);
        int m = t1.Height(), n = t2.Width(), k = t1.Width();
        float alpha = 1, beta = 0;

        for (uint32_t b = 0; b < output.Batch(); ++b)
        {
            uint32_t t1B = min(b, t1.Batch() - 1);
            uint32_t t2B = min(b, t2.Batch() - 1);

            for (uint32_t d = 0; d < t1.Depth(); ++d)
            {
                CUDA_CHECK(cublasSgemm_v2(
                    s_CublasHandle,
                    transposeT2 ? CUBLAS_OP_T : CUBLAS_OP_N,
                    transposeT1 ? CUBLAS_OP_T : CUBLAS_OP_N,
                    n,
                    m,
                    k,  // trick to convert row major to column major
                    &alpha,
                    t2.GetDevicePtr() + d * t2.GetShape().Dim0Dim1 + t2B * t2.BatchLength(),
                    n,
                    t1.GetDevicePtr() + d * t1.GetShape().Dim0Dim1 + t1B * t1.BatchLength(),
                    k,
                    &beta,
                    output.GetDevicePtr() + d * output.GetShape().Dim0Dim1 + b * output.BatchLength(),
                    n));
            }
        }
        cudaStreamSynchronize(0);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::MatMulBatched(bool transposeT1, bool transposeT2, const Tensor& t1, const Tensor& t2, Tensor& output) const
    {
        NVTXProfile nvtxProfile(__FUNCTION__, 0xFF004A7F);
        int m = t1.Height(), n = t2.Width(), k = t1.Width();

        size_t batches = output.Batch() * t1.Depth();
        vector<float> alphaArray(batches);
        fill(alphaArray.begin(), alphaArray.end(), 1.f);
        vector<float> betaArray(batches);
        fill(betaArray.begin(), betaArray.end(), 0.f);
        vector<const float*> t1List(batches);
        vector<const float*> t2List(batches);
        vector<float*> outputList(batches);

        for (uint32_t b = 0; b < output.Batch(); ++b)
        {
            uint32_t t1B = min(b, t1.Batch() - 1);
            uint32_t t2B = min(b, t2.Batch() - 1);

            for (uint32_t d = 0; d < t1.Depth(); ++d)
            {
                uint32_t idx = b * t1.Depth() + d;
                t1List[idx] = t1.GetDevicePtr() + d * t1.GetShape().Dim0Dim1 + t1B * t1.BatchLength();
                t2List[idx] = t2.GetDevicePtr() + d * t2.GetShape().Dim0Dim1 + t2B * t2.BatchLength();
                outputList[idx] = output.GetDevicePtr() + d * output.GetShape().Dim0Dim1 + b * output.BatchLength();
            }
        }

        float** devT1List = nullptr, **devT2List = nullptr, **devOutputList = nullptr;
        cudaMalloc(&devT1List, batches * sizeof(float*));
        cudaMalloc(&devT2List, batches * sizeof(float*));
        cudaMalloc(&devOutputList, batches * sizeof(float*));

        cudaMemcpy(devT1List, &t1List[0], batches * sizeof(float*), cudaMemcpyHostToDevice);
        cudaMemcpy(devT2List, &t2List[0], batches * sizeof(float*), cudaMemcpyHostToDevice);
        cudaMemcpy(devOutputList, &outputList[0], batches * sizeof(float*), cudaMemcpyHostToDevice);

        CUDA_CHECK(cublasSgemmBatched(
            s_CublasHandle,
            transposeT2 ? CUBLAS_OP_T : CUBLAS_OP_N,
            transposeT1 ? CUBLAS_OP_T : CUBLAS_OP_N,
            n,
            m,
            k,  // trick to convert row major to column major
            &alphaArray[0],
            devT2List,
            n,
            devT1List,
            k,
            &betaArray[0],
            devOutputList,
            n,
            (int)batches));
        cudaStreamSynchronize(0);

        cudaFree(devT1List);
        cudaFree(devT2List);
        cudaFree(devOutputList);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::MatMulStridedBatched(bool transposeT1, bool transposeT2, const Tensor& t1, const Tensor& t2, Tensor& output) const
    {
        NVTXProfile nvtxProfile(__FUNCTION__, 0xFF004A7F);
        int m = t1.Height(), n = t2.Width(), k = t1.Width();

        size_t batches = output.Depth() * output.Batch();
        vector<float> alphaArray(batches);
        fill(alphaArray.begin(), alphaArray.end(), 1.f);
        vector<float> betaArray(batches);
        fill(betaArray.begin(), betaArray.end(), 0.f);

        CUDA_CHECK(cublasSgemmStridedBatched(
            s_CublasHandle,
            transposeT2 ? CUBLAS_OP_T : CUBLAS_OP_N,
            transposeT1 ? CUBLAS_OP_T : CUBLAS_OP_N,
            n,
            m,
            k,  // trick to convert row major to column major
            &alphaArray[0],
            t2.GetDevicePtr(),
            n,
            t2.GetShape().Dim0Dim1,
            t1.GetDevicePtr(),
            k,
            t1.GetShape().Dim0Dim1,
            &betaArray[0],
            output.GetDevicePtr(),
            n,
            output.GetShape().Dim0Dim1,
            (int)batches));
        cudaStreamSynchronize(0);
    }

    //////////////////////////////////////////////////////////////////////////
    cudnnPoolingMode_t TensorOpGpu::GetCudnnPoolType(EPoolingMode mode)
    {
        if (mode == MaxPool)
            return CUDNN_POOLING_MAX;
        return CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
    }

    //////////////////////////////////////////////////////////////////////////
    cudnnBatchNormMode_t TensorOpGpu::GetCudnnBatchNormMode(EBatchNormMode mode)
    {
        if (mode == PerActivation)
            return CUDNN_BATCHNORM_PER_ACTIVATION;
        return CUDNN_BATCHNORM_SPATIAL;
    }

    //////////////////////////////////////////////////////////////////////////
    cudnnActivationMode_t TensorOpGpu::GetCudnnActivationMode(EActivation mode)
    {
        switch (mode)
        {
        case _Identity:
            return CUDNN_ACTIVATION_IDENTITY;
        case _Sigmoid:
            return CUDNN_ACTIVATION_SIGMOID;
        case _ReLU:
            return CUDNN_ACTIVATION_RELU;
        case _TanH:
            return CUDNN_ACTIVATION_TANH;
        case _ELU:
            return CUDNN_ACTIVATION_ELU;
        }
        NEURO_ASSERT(false, "Unsupported activation mode.");
        return CUDNN_ACTIVATION_IDENTITY;
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::GetKernelRunParams(int count, dim3& blocksDim, dim3& threadsDim, int maxThreads)
    {
        int blocks = GetBlocksNum(count, maxThreads);

        if (count <= maxThreads)
        {
            blocks = 1;
            maxThreads = count;
        }

        blocksDim = dim3(blocks, 1, 1);
        threadsDim = dim3(maxThreads, 1, 1);
    }

    //////////////////////////////////////////////////////////////////////////
    unsigned int nextPow2(unsigned int x)
    {
        --x;
        x |= x >> 1;
        x |= x >> 2;
        x |= x >> 4;
        x |= x >> 8;
        x |= x >> 16;
        return ++x;
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::GetGlobalSumKernelRunParams(int count, dim3& blocksDim, dim3& threadsDim, int maxThreads)
    {
        int whichKernel = 3;
        int blocks, threads;

        if (whichKernel < 3)
        {
            threads = (count < maxThreads) ? nextPow2(count) : maxThreads;
            blocks = (count + threads - 1) / threads;
        }
        else
        {
            threads = (count < maxThreads * 2) ? nextPow2((count + 1) / 2) : maxThreads;
            blocks = (count + (threads * 2 - 1)) / (threads * 2);
        }

        if ((float)threads*blocks > (float)s_CudaDevProp.maxGridSize[0] * s_CudaDevProp.maxThreadsPerBlock)
        {
            printf("n is too large, please choose a smaller number!\n");
        }

        if (blocks > s_CudaDevProp.maxGridSize[0])
        {
            printf("Grid size <%d> exceeds the device capability <%d>, set block size as %d (original %d)\n",
                blocks, s_CudaDevProp.maxGridSize[0], threads * 2, threads);

            blocks /= 2;
            threads *= 2;
        }

        blocksDim = dim3(blocks, 1, 1);
        threadsDim = dim3(threads, 1, 1);
    }

    //////////////////////////////////////////////////////////////////////////
    int TensorOpGpu::GetBlocksNum(int count, int threadsPerBlock)
    {
        return (int)ceil(count / (float)threadsPerBlock);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::CudnnLog(cudnnSeverity_t sev, void *udata, const cudnnDebug_t *dbg, const char *msg)
    {
        OutputDebugString(msg);
        OutputDebugString("\n");
    }
}
