﻿#include <sstream>
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

    static const int INNER_KERNEL_LOOP_LENGTH = 1; // for simple per-element kernels

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

                //size_t reservedBytes = (size_t)(freeBytes * 0.85);
                //CUDA_CHECK(MemoryManager::Default().Reserve(reservedBytes));

                stringstream ss;
                ss << "GPU >> " << s_CudaDevProp.name << " threads_per_block=" << s_CudaDevProp.maxThreadsPerBlock << " available/total_memory=" << freeBytes/(1024*1024) << "/" << totalBytes/(1024*1024) << "MB\n";
                //ss << "Reserved memory: " << reservedBytes/(1024*1024) << "MB\n";
                OutputDebugString(ss.str().c_str());
            }
        }
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::Zero(Tensor& input) const
    {
        input.OverrideDevice();
        CUDA_CHECK(cudaMemset(input.GetDevicePtr(), 0, input.Length() * sizeof(float)));
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::One(Tensor& input) const
    {
        float one = 1.f;
        unsigned int* oneBits = reinterpret_cast<unsigned int*>(&one);
        input.OverrideDevice();
        cuMemsetD32(CUdeviceptr(input.GetDevicePtr()), *oneBits, input.Length());
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::Add(float alpha, const Tensor& t1, float beta, const Tensor& t2, Tensor& output) const
    {
        t1.CopyToDevice();
        t2.CopyToDevice();
        output.OverrideDevice();

        if (!t1.SameDimensionsExceptBatches(t2))
        {
            dim3 blocks, threads;
            GetKernelRunParams(output.Length(), blocks, threads, s_CudaDevProp.maxThreadsPerBlock);

            if (t2.SameDimensionsExceptBatches(output))
            {
                if (t1.SameDimensionsOrOne(output))
                {
                    if (t2.GetDevicePtr() != output.GetDevicePtr())
                        t2.CopyTo(output);

                    cudnnTensorDescriptor_t biasDesc; cudnnCreateTensorDescriptor(&biasDesc);
                    cudnnTensorDescriptor_t outputDesc; cudnnCreateTensorDescriptor(&outputDesc);
                    cudnnSetTensor4dDescriptor(biasDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, t1.GetShape().Dimensions[3], t1.GetShape().Dimensions[2], t1.GetShape().Dimensions[1], t1.GetShape().Dimensions[0]);
                    cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, output.GetShape().Dimensions[3], output.GetShape().Dimensions[2], output.GetShape().Dimensions[1], output.GetShape().Dimensions[0]);

                    CUDA_CHECK(cudnnAddTensor(s_CudnnHandle, &beta, biasDesc, t1.GetDevicePtr(), &alpha, outputDesc, output.GetDevicePtr()));
                    return;
                }

                return CudaKernels::AddBroadcast(
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
            }
            else if (t1.SameDimensionsExceptBatches(output))
            {
                if (t2.SameDimensionsOrOne(output))
                {
                    if (t1.GetDevicePtr() != output.GetDevicePtr())
                        t1.CopyTo(output);

                    cudnnTensorDescriptor_t biasDesc; cudnnCreateTensorDescriptor(&biasDesc);
                    cudnnTensorDescriptor_t outputDesc; cudnnCreateTensorDescriptor(&outputDesc);
                    cudnnSetTensor4dDescriptor(biasDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, t2.GetShape().Dimensions[3], t2.GetShape().Dimensions[2], t2.GetShape().Dimensions[1], t2.GetShape().Dimensions[0]);
                    cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, output.GetShape().Dimensions[3], output.GetShape().Dimensions[2], output.GetShape().Dimensions[1], output.GetShape().Dimensions[0]);

                    CUDA_CHECK(cudnnAddTensor(s_CudnnHandle, &beta, biasDesc, t2.GetDevicePtr(), &alpha, outputDesc, output.GetDevicePtr()));
                    return;
                }

                return CudaKernels::AddBroadcast(
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
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::MatMul(bool transposeT1, bool transposeT2, const Tensor& t1, const Tensor& t2, Tensor& output) const
    {
        t1.CopyToDevice();
        t2.CopyToDevice();
        output.OverrideDevice();
        output.Zero();

        if (t1.Depth() == t2.Depth() && t1.Batch() == t2.Batch())
            MulStridedBatched(transposeT1, transposeT2, t1, t2, output);
        else if (t1.Depth() * output.Batch() > 48)
            MulBatched(transposeT1, transposeT2, t1, t2, output);
        else
            MulGeneric(transposeT1, transposeT2, t1, t2, output);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::MulElem(const Tensor& t1, const Tensor& t2, Tensor& output) const
    {
        t1.CopyToDevice();
        t2.CopyToDevice();
        output.OverrideDevice();

        dim3 blocks, threads;

        if (t1.GetShape() == t2.GetShape())
        {
            GetKernelRunParams(max(t1.Length() / INNER_KERNEL_LOOP_LENGTH, 1), blocks, threads, s_CudaDevProp.maxThreadsPerBlock);
            CudaKernels::MulElem(blocks, threads, t1.Length(), t1.GetDevicePtr(), t2.GetDevicePtr(), output.GetDevicePtr(), INNER_KERNEL_LOOP_LENGTH);
        }
        else
        {
            dim3 blocks, threads;
            GetKernelRunParams(output.Length(), blocks, threads, s_CudaDevProp.maxThreadsPerBlock);

            if (t2.SameDimensionsExceptBatches(output))
            {
                return CudaKernels::MulElemBroadcast(
                    blocks,
                    threads,
                    t2.GetDevicePtr(),
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
            }
            else if (t1.SameDimensionsExceptBatches(output))
            {
                return CudaKernels::MulElemBroadcast(
                    blocks,
                    threads,
                    t1.GetDevicePtr(),
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
            }
            else
                __super::MulElem(t1, t2, output);
        }        
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::Div(const Tensor& t1, const Tensor& t2, Tensor& output) const
    {
        t1.CopyToDevice();
        t2.CopyToDevice();
        output.OverrideDevice();

        dim3 blocks, threads;

        if (t1.GetShape() == t2.GetShape())
        {
            GetKernelRunParams(max(t1.Length() / INNER_KERNEL_LOOP_LENGTH, 1), blocks, threads, s_CudaDevProp.maxThreadsPerBlock);
            CudaKernels::Div(blocks, threads, t1.Length(), t1.GetDevicePtr(), t2.GetDevicePtr(), output.GetDevicePtr(), INNER_KERNEL_LOOP_LENGTH);
        }
        else
        {
            GetKernelRunParams(output.Length(), blocks, threads, s_CudaDevProp.maxThreadsPerBlock / 2);

            return CudaKernels::DivBroadcast(
                blocks,
                threads,
                t1.GetDevicePtr(),
                t1.Width(),
                t1.Height(),
                t1.Depth(),
                t1.Batch(),
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
        }
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::Mul(const Tensor& input, float v, Tensor& output) const
    {
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
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::Div(const Tensor& input, float v, Tensor& output) const
    {
        Mul(input, 1.f / v, output);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::Pow(const Tensor& input, float power, Tensor& output) const
    {
        dim3 blocks, threads;
        GetKernelRunParams(max((int)input.Length() / INNER_KERNEL_LOOP_LENGTH, 1), blocks, threads, s_CudaDevProp.maxThreadsPerBlock);
        input.CopyToDevice();
        output.OverrideDevice();

        CudaKernels::Pow(blocks, threads, input.Length(), input.GetDevicePtr(), power, output.GetDevicePtr(), INNER_KERNEL_LOOP_LENGTH);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::PowGradient(const Tensor& input, float power, const Tensor& outputGradient, Tensor& inputGradient) const
    {
        dim3 blocks, threads;
        GetKernelRunParams(max((int)input.Length() / INNER_KERNEL_LOOP_LENGTH, 1), blocks, threads, s_CudaDevProp.maxThreadsPerBlock);
        input.CopyToDevice();
        outputGradient.CopyToDevice();
        inputGradient.OverrideDevice();

        CudaKernels::PowGradient(blocks, threads, input.Length(), input.GetDevicePtr(), power, outputGradient.GetDevicePtr(), inputGradient.GetDevicePtr(), INNER_KERNEL_LOOP_LENGTH);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::Negate(const Tensor& input, Tensor& output) const
    {
        dim3 blocks, threads;
        GetKernelRunParams(max((int)input.Length() / INNER_KERNEL_LOOP_LENGTH, 1), blocks, threads, s_CudaDevProp.maxThreadsPerBlock);
        input.CopyToDevice();
        output.OverrideDevice();

        CudaKernels::Negate(blocks, threads, input.Length(), input.GetDevicePtr(), output.GetDevicePtr(), INNER_KERNEL_LOOP_LENGTH);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::Inverse(const Tensor& input, Tensor& output) const
    {
        dim3 blocks, threads;
        GetKernelRunParams(max((int)input.Length() / INNER_KERNEL_LOOP_LENGTH, 1), blocks, threads, s_CudaDevProp.maxThreadsPerBlock);
        input.CopyToDevice();
        output.OverrideDevice();

        CudaKernels::Inverse(blocks, threads, input.Length(), input.GetDevicePtr(), output.GetDevicePtr(), INNER_KERNEL_LOOP_LENGTH);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::Add(const Tensor& input, float v, Tensor& output) const
    {
        dim3 blocks, threads;
        GetKernelRunParams(max((int)input.Length() / INNER_KERNEL_LOOP_LENGTH, 1), blocks, threads, s_CudaDevProp.maxThreadsPerBlock);
        input.CopyToDevice();
        output.OverrideDevice();

        CudaKernels::Add(blocks, threads, input.Length(), input.GetDevicePtr(), v, output.GetDevicePtr(), INNER_KERNEL_LOOP_LENGTH);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::Transpose(const Tensor& input, Tensor& output) const
    {
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
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::Conv2D(const Tensor& input, const Tensor& kernels, uint32_t stride, uint32_t paddingX, uint32_t paddingY, EDataFormat dataFormat, Tensor& output) const
    {
        input.CopyToDevice();
        kernels.CopyToDevice();
        output.OverrideDevice();
        output.Zero();

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

        DeviceMemoryManager::Default().Free(workspacePtr);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::Conv2DBiasActivation(const Tensor& input, const Tensor& kernels, uint32_t stride, uint32_t paddingX, uint32_t paddingY, const Tensor& bias, EActivation activation, float activationAlpha, Tensor& output)
    {
        input.CopyToDevice();
        kernels.CopyToDevice();
        bias.CopyToDevice();
        output.OverrideDevice();
        output.Zero();

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

        DeviceMemoryManager::Default().Free(workspacePtr);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::Conv2DBiasGradient(const Tensor& gradient, Tensor& biasGradient)
    {
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
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::Conv2DInputGradient(const Tensor& gradient, const Tensor& kernels, uint32_t stride, uint32_t paddingX, uint32_t paddingY, EDataFormat dataFormat, Tensor& inputGradient) const
    {
        if (dataFormat == NHWC) // CuDNN doesn't support gradients for NHWC format
            return __super::Conv2DInputGradient(gradient, kernels, stride, paddingX, paddingY, dataFormat, inputGradient);

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

        DeviceMemoryManager::Default().Free(workspacePtr);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::Conv2DKernelsGradient(const Tensor& input, const Tensor& gradient, uint32_t stride, uint32_t paddingX, uint32_t paddingY, EDataFormat dataFormat, Tensor& kernelsGradient) const
    {
        if (dataFormat == NHWC) // CuDNN doesn't support gradients for NHWC format
            return __super::Conv2DKernelsGradient(input, gradient, stride, paddingX, paddingY, dataFormat, kernelsGradient);

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

        DeviceMemoryManager::Default().Free(workspacePtr);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::Pool2D(const Tensor& input, uint32_t filterSize, uint32_t stride, EPoolingMode type, uint32_t paddingX, uint32_t paddingY, EDataFormat dataFormat, Tensor& output) const
    {
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
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::Pool2DGradient(const Tensor& output, const Tensor& input, const Tensor& outputGradient, uint32_t filterSize, uint32_t stride, EPoolingMode type, uint32_t paddingX, uint32_t paddingY, EDataFormat dataFormat, Tensor& inputGradient) const
    {
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
    }

    ////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::UpSample2D(const Tensor& input, uint32_t scaleFactor, Tensor& output) const
    {
        dim3 blocks, threads;
        GetKernelRunParams(input.Length(), blocks, threads, s_CudaDevProp.maxThreadsPerBlock);
        input.CopyToDevice();
        output.OverrideDevice();

        CudaKernels::UpSample2D(blocks, threads, input.GetDevicePtr(), input.Width(), input.Height(), input.Depth(), input.Batch(), scaleFactor, output.GetDevicePtr());
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::UpSample2DGradient(const Tensor& outputGradient, uint32_t scaleFactor, Tensor& inputGradient) const
    {
        dim3 blocks, threads;
        GetKernelRunParams(inputGradient.Length(), blocks, threads, s_CudaDevProp.maxThreadsPerBlock);
        outputGradient.CopyToDevice();
        inputGradient.OverrideDevice();
        inputGradient.Zero();
        
        CudaKernels::UpSample2DGradient(blocks, threads, outputGradient.GetDevicePtr(), scaleFactor, inputGradient.GetDevicePtr(), inputGradient.Width(), inputGradient.Height(), inputGradient.Depth(), inputGradient.Batch());
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::BatchNormalization(const Tensor& input, EBatchNormMode mode, const Tensor& gamma, const Tensor& beta, float epsilon, const Tensor* runningMean, const Tensor* runningVar, Tensor& output) const
    {
        if (mode == Instance)
            return __super::BatchNormalization(input, mode, gamma, beta, epsilon, runningMean, runningVar, output);

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
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::BatchNormalizationTrain(const Tensor& input, EBatchNormMode mode, const Tensor& gamma, const Tensor& beta, float momentum, float epsilon, Tensor* runningMean, Tensor* runningVar, Tensor& saveMean, Tensor& saveInvVariance, Tensor& output) const
    {
        if (mode == Instance)
            return __super::BatchNormalizationTrain(input, mode, gamma, beta, momentum, epsilon, runningMean, runningVar, saveMean, saveInvVariance, output);

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
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::BatchNormalizationGradient(const Tensor& input, EBatchNormMode mode, const Tensor& gamma, float epsilon, const Tensor& outputGradient, const Tensor& savedMean, const Tensor& savedInvVariance, Tensor& gammaGradient, Tensor& betaGradient, bool trainable, Tensor& inputGradient) const
    {
        if (mode == Instance)
            return __super::BatchNormalizationGradient(input, mode, gamma, epsilon, outputGradient, savedMean, savedInvVariance, gammaGradient, betaGradient, trainable, inputGradient);

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
    }

    //////////////////////////////////////////////////////////////////////////
    //void TensorOpGpu::Dropout(const Tensor& input, float prob, Tensor& saveMask, void** states, Tensor& output)
    //{
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
    //}

    //////////////////////////////////////////////////////////////////////////////
    //void TensorOpGpu::DropoutGradient(const Tensor& outputGradient, const Tensor& savedMask, Tensor& inputGradient)
    //{
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
        dim3 blocks, threads;
        GetKernelRunParams(input.Length(), blocks, threads, s_CudaDevProp.maxThreadsPerBlock);
        input.CopyToDevice();
        output.OverrideDevice();

        CudaKernels::LeakyReLU(blocks, threads, input.Length(), input.GetDevicePtr(), alpha, output.GetDevicePtr());
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::LeakyReLUGradient(const Tensor& output, const Tensor& outputGradient, float alpha, Tensor& inputGradient) const
    {
        dim3 blocks, threads;
        GetKernelRunParams(output.Length(), blocks, threads, s_CudaDevProp.maxThreadsPerBlock);
        output.CopyToDevice();
        outputGradient.CopyToDevice();
        inputGradient.OverrideDevice();
        inputGradient.Zero();

        CudaKernels::LeakyReLUGradient(blocks, threads, output.Length(), output.GetDevicePtr(), outputGradient.GetDevicePtr(), alpha, inputGradient.GetDevicePtr());
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::Softmax(const Tensor& input, Tensor& output) const
    {
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
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::SoftmaxGradient(const Tensor& output, const Tensor& outputGradient, Tensor& inputGradient) const
    {
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
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::Sum(const Tensor& input, EAxis axis, Tensor& output) const
    {
        if (axis == GlobalAxis && s_CudaDevProp.maxThreadsPerBlock < 1024)
            __super::Sum(input, axis, output);


        input.CopyToDevice();
        output.OverrideDevice();
            
        const int THREADS_PER_BLOCK = 1024;

        auto globalSumKernelCall = [&](uint32_t n, const float* inputDevPtr, float* outputDevPtr)
        {
            dim3 blocks, threads;
            GetGlobalSumKernelRunParams(n, blocks, threads, THREADS_PER_BLOCK);
            return CudaKernels::Sum(blocks, threads, inputDevPtr, n, 1, 1, 1, GlobalAxis, outputDevPtr);
        };

        auto sumGlobalInternal = [&](uint32_t inputLen, const float* inputDevPtr, float* outputDevPtr)
        {
            dim3 blocks, threads;
            GetGlobalSumKernelRunParams(inputLen, blocks, threads, THREADS_PER_BLOCK);
            
            if (blocks.x == 1)
                return CudaKernels::Sum(blocks, threads, inputDevPtr, inputLen, 1, 1, 1, GlobalAxis, outputDevPtr);

            void* tempOutput;
            int tempOutputLen = blocks.x;
            DeviceMemoryManager::Default().Allocate(&tempOutput, blocks.x * sizeof(float));

            CudaKernels::Sum(blocks, threads, inputDevPtr, inputLen, 1, 1, 1, GlobalAxis, (float*)tempOutput);

            while(true)
            {
                int n = blocks.x;
                GetGlobalSumKernelRunParams(n, blocks, threads, THREADS_PER_BLOCK);

                //for debug
                /*vector<float> tmp(blocks.x);
                cudaMemcpy(&tmp[0], tempOutput, tmp.size() * sizeof(float), cudaMemcpyDeviceToHost);*/

                CudaKernels::Sum(blocks, threads, (float*)tempOutput, n, 1, 1, 1, GlobalAxis, (float*)tempOutput);
                
                if (blocks.x == 1)
                    break;
            }

            cudaMemcpy(outputDevPtr, tempOutput, sizeof(float), cudaMemcpyDeviceToDevice);
            DeviceMemoryManager::Default().Free(tempOutput);
            return;
        };
        
        if (axis == GlobalAxis)
        {
            return sumGlobalInternal(input.Length(), input.GetDevicePtr(), output.GetDevicePtr());
        }

        if (axis == WidthAxis || axis == _01Axes || axis == _012Axes)
        {
            uint32_t fakeWidth = input.Width();
            if (axis == _01Axes)
                fakeWidth = input.Width() * input.Height();
            else if (axis == _012Axes)
                fakeWidth = input.Width() * input.Height() * input.Depth();

            dim3 blocksPerRow, threads;
            GetGlobalSumKernelRunParams(fakeWidth, blocksPerRow, threads, THREADS_PER_BLOCK);

            // block x - one per output element, y - number or blocks required to compute a single row sum
            dim3 blocks{output.Length(), blocksPerRow.x};

            if (blocksPerRow.x == 1)
                return CudaKernels::Sum(blocks, threads, input.GetDevicePtr(), fakeWidth, 1, 1, 1, WidthAxis, output.GetDevicePtr());

            void* tempOutput;
            int tempOutputLen = blocks.x;
            DeviceMemoryManager::Default().Allocate(&tempOutput, blocks.x * blocks.y * sizeof(float));

            CudaKernels::Sum(blocks, threads, input.GetDevicePtr(), fakeWidth, 1, 1, 1, WidthAxis, (float*)tempOutput);

            //for debug
            /*vector<float> tmp(blocks.x * blocks.y);
            cudaMemcpy(&tmp[0], tempOutput, tmp.size() * sizeof(float), cudaMemcpyDeviceToHost);*/

            while (true)
            {
                fakeWidth = blocks.y; // new width
                GetGlobalSumKernelRunParams(fakeWidth, blocksPerRow, threads, THREADS_PER_BLOCK);

                blocks = { output.Length(), blocksPerRow.x };
                CudaKernels::Sum(blocks, threads, (float*)tempOutput, fakeWidth, 1, 1, 1, WidthAxis, (float*)tempOutput);

                /*tmp.clear();
                tmp.resize(blocks.x * blocks.y);
                cudaMemcpy(&tmp[0], tempOutput, tmp.size() * sizeof(float), cudaMemcpyDeviceToHost);*/

                if (blocksPerRow.x == 1)
                    break;
            }

            cudaMemcpy(output.GetDevicePtr(), tempOutput, output.Length() * sizeof(float), cudaMemcpyDeviceToDevice);
            DeviceMemoryManager::Default().Free(tempOutput);
            return;
        }

        //if (axis == _013Axes)
        //{
        //    Tensor tmp(Shape(1,1,input.Depth())); //temporary tensor to store single batch _01Axis sum
        //    tmp.OverrideDevice();

        //    for (uint32_t n = 0; n < input.Batch(); ++n)
        //    {
        //        for (uint32_t d = 0; d < input.Depth(); ++d)
        //            sumGlobalInternal(input.GetDevicePtr() + d * input.Stride(2) + n * input.Stride(3), input.Stride(2), tmp.GetDevicePtr() + d); // perform _01Axis sum

        //        Add(1, tmp, 1, output, output);
        //    }
        //    return;
        //}

        if (axis == _013Axes)
        {
            Tensor tmp(Shape(1, 1, input.Depth(), input.Batch()));
            tmp.OverrideDevice();

            Sum(input, _01Axes, tmp);
            Sum(tmp, BatchAxis, output);
            return;
        }

        if (axis != EAxis::BatchAxis)
        {
            dim3 blocks, threads;
            GetKernelRunParams(output.Length(), blocks, threads, s_CudaDevProp.maxThreadsPerBlock);
            return CudaKernels::Sum(blocks, threads, input.GetDevicePtr(), input.Width(), input.Height(), input.Depth(), input.Batch(), axis, output.GetDevicePtr());
        }

        output.Zero();

        float alpha = 1, beta = 1;
        for (uint32_t n = 0; n < input.Batch(); ++n)
        {
            CUDA_CHECK(cublasSgeam(
                s_CublasHandle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                input.BatchLength(),
                1,
                &alpha,
                input.GetDevicePtr() + n * input.BatchLength(),
                input.BatchLength(),
                &beta,
                output.GetDevicePtr(),
                output.BatchLength(),
                output.GetDevicePtr(),
                output.BatchLength()));
        }
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::AdamStep(Tensor& parameter, const Tensor& gradient, Tensor& mGrad, Tensor& vGrad, /*float batchSize, */float lr, float beta1, float beta2, float epsilon) const
    {
        parameter.CopyToDevice();
        gradient.CopyToDevice();
        mGrad.CopyToDevice();
        vGrad.CopyToDevice();

        dim3 blocks, threads;
        GetKernelRunParams(parameter.Length(), blocks, threads, s_CudaDevProp.maxThreadsPerBlock);
        
        CudaKernels::AdamStep(blocks, threads, parameter.Length(), parameter.GetDevicePtr(), gradient.GetDevicePtr(), mGrad.GetDevicePtr(), vGrad.GetDevicePtr(), /*batchSize, */lr, beta1, beta2, epsilon);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::SgdStep(Tensor& parameter, const Tensor& gradient, /*float batchSize, */float lr) const
    {
        parameter.CopyToDevice();
        gradient.CopyToDevice();

        dim3 blocks, threads;
        GetKernelRunParams(parameter.Length(), blocks, threads, s_CudaDevProp.maxThreadsPerBlock);

        CudaKernels::SgdStep(blocks, threads, parameter.Length(), parameter.GetDevicePtr(), gradient.GetDevicePtr(), /*batchSize, */lr);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::Activation(const cudnnActivationMode_t& activationMode, const Tensor& input, Tensor& output, float coeff) const
    {
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
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::ActivationGradient(const cudnnActivationMode_t& activationMode, const Tensor& output, const Tensor& outputGradient, Tensor& inputGradient, float coeff) const
    {
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
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::MulGeneric(bool transposeT1, bool transposeT2, const Tensor& t1, const Tensor& t2, Tensor& output) const
    {
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
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::MulBatched(bool transposeT1, bool transposeT2, const Tensor& t1, const Tensor& t2, Tensor& output) const
    {
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

        cudaFree(devT1List);
        cudaFree(devT2List);
        cudaFree(devOutputList);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::MulStridedBatched(bool transposeT1, bool transposeT2, const Tensor& t1, const Tensor& t2, Tensor& output) const
    {
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

        /*if (whichKernel == 6)
        {
            blocks = min(maxBlocks, blocks);
        }*/

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
