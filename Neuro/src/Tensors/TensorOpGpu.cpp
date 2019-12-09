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
    curandGenerator_t TensorOpGpu::s_CurandGenerator = nullptr;

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
                curandCreateGenerator(&s_CurandGenerator, CURAND_RNG_PSEUDO_DEFAULT);

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
        DeviceMemoryManager::Default().ScheduleFree(ptr);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::Zero(Tensor& input) const
    {
        NVTXProfile nvtxProfile(__FUNCTION__, 0xFF004A7F);
        input.OverrideDevice();

        float zero = 0.f;
        CUDA_CHECK(cudnnSetTensor(s_CudnnHandle, input.DeviceDesc(), input.GetDevicePtr(), &zero));
        cudaStreamSynchronize(0);
        //CUDA_CHECK(cudaMemsetAsync(input.GetDevicePtr(), 0, input.Length() * sizeof(float)));
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::One(Tensor& input) const
    {
        NVTXProfile nvtxProfile(__FUNCTION__, 0xFF004A7F);
        input.OverrideDevice();

        float one = 1.f;
        CUDA_CHECK(cudnnSetTensor(s_CudnnHandle, input.DeviceDesc(), input.GetDevicePtr(), &one));
        cudaStreamSynchronize(0);
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
            if (t1.SameDimensionsOrOne(t2))
            {
                cudnnOpTensorDescriptor_t addDesc; cudnnCreateOpTensorDescriptor(&addDesc);
                cudnnSetOpTensorDescriptor(addDesc, CUDNN_OP_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN);
                
                float beta2 = 0.f;
                CUDA_CHECK(cudnnOpTensor(s_CudnnHandle, addDesc, &alpha, t1.DeviceDesc(), t1.GetDevicePtr(), &beta, t2.DeviceDesc(), t2.GetDevicePtr(), &beta2, output.DeviceDesc(), output.GetDevicePtr()));
                cudaStreamSynchronize(0);

                cudnnDestroyOpTensorDescriptor(addDesc);
                return;
            }
            if (t2.SameDimensionsOrOne(t1))
            {
                cudnnOpTensorDescriptor_t addDesc; cudnnCreateOpTensorDescriptor(&addDesc);
                cudnnSetOpTensorDescriptor(addDesc, CUDNN_OP_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN);

                float beta2 = 0.f;
                CUDA_CHECK(cudnnOpTensor(s_CudnnHandle, addDesc, &beta, t2.DeviceDesc(), t2.GetDevicePtr(), &alpha, t1.DeviceDesc(), t1.GetDevicePtr(), &beta2, output.DeviceDesc(), output.GetDevicePtr()));
                cudaStreamSynchronize(0);

                cudnnDestroyOpTensorDescriptor(addDesc);
                return;
            }

            dim3 blocks, threads;
            GetKernelRunParamsForSequence(output.Length(), blocks, threads, 128);
            CudaKernels::AddBroadcast(
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
            return;
        }

        if (t2.Batch() == t1.Batch())
        {
            if (output.GetDevicePtr() == t1.GetDevicePtr())
            {
                CUDA_CHECK(cudnnAddTensor(s_CudnnHandle, &beta, t2.DeviceDesc(), t2.GetDevicePtr(), &alpha, output.DeviceDesc(), output.GetDevicePtr()));
                cudaStreamSynchronize(0);
                return;
            }

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
            cudnnSetOpTensorDescriptor(mulDesc, CUDNN_OP_TENSOR_MUL, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN);

            float beta2 = 0.f;
            if (t1.SameDimensionsOrOne(t2))
                CUDA_CHECK(cudnnOpTensor(s_CudnnHandle, mulDesc, &alpha, t1.DeviceDesc(), t1.GetDevicePtr(), &beta, t2.DeviceDesc(), t2.GetDevicePtr(), &beta2, output.DeviceDesc(), output.GetDevicePtr()));
            else
                CUDA_CHECK(cudnnOpTensor(s_CudnnHandle, mulDesc, &beta, t2.DeviceDesc(), t2.GetDevicePtr(), &alpha, t1.DeviceDesc(), t1.GetDevicePtr(), &beta2, output.DeviceDesc(), output.GetDevicePtr()));
            cudaStreamSynchronize(0);

            cudnnDestroyOpTensorDescriptor(mulDesc);
            return;
        }

        dim3 blocks, threads;
        GetKernelRunParamsForSequence(output.Length(), blocks, threads, 128);
        CudaKernels::MulBroadcast(
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

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::Scale(Tensor& input, float v) const
    {
        NVTXProfile nvtxProfile(__FUNCTION__, 0xFF004A7F);
        input.CopyToDevice();

        CUDA_CHECK(cudnnScaleTensor(s_CudnnHandle, input.DeviceDesc(), input.GetDevicePtr(), &v));
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
            GetKernelRunParamsForSequence(t1.Length(), blocks, threads, 128);
            CudaKernels::Div(blocks, threads, t1.Length(), alpha, t1.GetDevicePtr(), beta, t2.GetDevicePtr(), output.GetDevicePtr());
            cudaStreamSynchronize(0);
        }
        else
        {
            GetKernelRunParamsForSequence(output.Length(), blocks, threads, 128);
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
            cudnnSetOpTensorDescriptor(sqrDesc, CUDNN_OP_TENSOR_MUL, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN);

            float alpha1 = 1.f, alpha2 = 1.f, beta = 0.f;
            CUDA_CHECK(cudnnOpTensor(s_CudnnHandle, sqrDesc, &alpha1, input.DeviceDesc(), input.GetDevicePtr(), &alpha2, input.DeviceDesc(), input.GetDevicePtr(), &beta, output.DeviceDesc(), output.GetDevicePtr()));
            cudaStreamSynchronize(0);

            cudnnDestroyOpTensorDescriptor(sqrDesc);
            return;
        }

        dim3 blocks, threads;
        GetKernelRunParamsForSequence(input.Length(), blocks, threads, 128);
        CudaKernels::Pow(blocks, threads, input.Length(), input.GetDevicePtr(), power, output.GetDevicePtr());
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
        GetKernelRunParamsForSequence(input.Length(), blocks, threads, 128);
        CudaKernels::PowGradient(blocks, threads, input.Length(), input.GetDevicePtr(), power, outputGradient.GetDevicePtr(), inputGradient.GetDevicePtr());
        cudaStreamSynchronize(0);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::Abs(const Tensor& input, Tensor& output) const
    {
        NVTXProfile nvtxProfile(__FUNCTION__, 0xFF004A7F);
        input.CopyToDevice();
        output.OverrideDevice();

        dim3 blocks, threads;
        GetKernelRunParamsForSequence(input.Length(), blocks, threads, 128);
        CudaKernels::Abs(blocks, threads, input.Length(), input.GetDevicePtr(), output.GetDevicePtr());
        cudaStreamSynchronize(0);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::AbsGradient(const Tensor& input, const Tensor& outputGradient, Tensor& inputGradient) const
    {
        NVTXProfile nvtxProfile(__FUNCTION__, 0xFF004A7F);
        input.CopyToDevice();
        outputGradient.CopyToDevice();
        inputGradient.OverrideDevice();

        dim3 blocks, threads;
        GetKernelRunParamsForSequence(input.Length(), blocks, threads, 128);
        CudaKernels::AbsGradient(blocks, threads, input.Length(), input.GetDevicePtr(), outputGradient.GetDevicePtr(), inputGradient.GetDevicePtr());
        cudaStreamSynchronize(0);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::Sqrt(const Tensor& input, Tensor& output) const
    {
        NVTXProfile nvtxProfile(__FUNCTION__, 0xFF004A7F);
        input.CopyToDevice();
        output.OverrideDevice();

        cudnnOpTensorDescriptor_t sqrDesc; cudnnCreateOpTensorDescriptor(&sqrDesc);
        cudnnSetOpTensorDescriptor(sqrDesc, CUDNN_OP_TENSOR_SQRT, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN);

        float alpha1 = 1, alpha2 = 0, beta = 0;
        CUDA_CHECK(cudnnOpTensor(s_CudnnHandle, sqrDesc, &alpha1, input.DeviceDesc(), input.GetDevicePtr(), &alpha2, input.DeviceDesc(), input.GetDevicePtr(), &beta, output.DeviceDesc(), output.GetDevicePtr()));
        cudaStreamSynchronize(0);

        cudnnDestroyOpTensorDescriptor(sqrDesc);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::Log(const Tensor& input, Tensor& output) const
    {
        NVTXProfile nvtxProfile(__FUNCTION__, 0xFF004A7F);
        input.CopyToDevice();
        output.OverrideDevice();

        dim3 blocks, threads;
        GetKernelRunParamsForSequence(input.Length(), blocks, threads, 128);
        CudaKernels::Log(blocks, threads, input.Length(), input.GetDevicePtr(), output.GetDevicePtr());
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

        dim3 blocks, threads;
        GetKernelRunParamsForSequence(input.Length(), blocks, threads, 128);
        CudaKernels::Inverse(blocks, threads, input.Length(), input.GetDevicePtr(), alpha, output.GetDevicePtr());
        cudaStreamSynchronize(0);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::Clip(const Tensor& input, float min, float max, Tensor& output) const
    {
        NVTXProfile nvtxProfile(__FUNCTION__, 0xFF004A7F);
        input.CopyToDevice();
        output.OverrideDevice();

        dim3 blocks, threads;
        GetKernelRunParamsForSequence(input.Length(), blocks, threads, 128);
        CudaKernels::Clip(blocks, threads, input.Length(), input.GetDevicePtr(), min, max, output.GetDevicePtr());
        cudaStreamSynchronize(0);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::ClipGradient(const Tensor& input, float min, float max, const Tensor& outputGradient, Tensor& inputGradient) const
    {
        NVTXProfile nvtxProfile(__FUNCTION__, 0xFF004A7F);
        input.CopyToDevice();
        outputGradient.CopyToDevice();
        inputGradient.OverrideDevice();

        dim3 blocks, threads;
        GetKernelRunParamsForSequence(input.Length(), blocks, threads, 128);
        CudaKernels::ClipGradient(blocks, threads, input.Length(), input.GetDevicePtr(), min, max, outputGradient.GetDevicePtr(), inputGradient.GetDevicePtr());
        cudaStreamSynchronize(0);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::Add(const Tensor& input, float v, Tensor& output) const
    {
        NVTXProfile nvtxProfile(__FUNCTION__, 0xFF004A7F);
        input.CopyToDevice();
        output.OverrideDevice();

        dim3 blocks, threads;
        GetKernelRunParamsForSequence(input.Length(), blocks, threads, 128);
        CudaKernels::Add(blocks, threads, input.Length(), input.GetDevicePtr(), v, output.GetDevicePtr());
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
    void TensorOpGpu::Transpose(const Tensor& input, const vector<EAxis>& permutation, Tensor& output) const
    {
        NVTXProfile nvtxProfile(__FUNCTION__, 0xFF004A7F);
        input.CopyToDevice();
        output.OverrideDevice();

        dim3 blocks, threads;
        GetKernelRunParamsForSequence(input.Length(), blocks, threads, 128);
        CudaKernels::Transpose(blocks, threads, input.Length(), input.GetDevicePtr(), permutation[0], permutation[1], permutation[2], permutation[3], input.Stride(0), input.Stride(1), input.Stride(2), input.Stride(3), output.GetDevicePtr(), output.Stride(1), output.Stride(2), output.Stride(3));
        cudaStreamSynchronize(0);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::ConstantPad2D(const Tensor& input, uint32_t left, uint32_t right, uint32_t top, uint32_t bottom, float value, Tensor& output) const
    {
        NVTXProfile nvtxProfile(__FUNCTION__, 0xFF004A7F);
        input.CopyToDevice();
        output.OverrideDevice();

        dim3 blocks, threads;
        GetKernelRunParamsForSequence(output.Length(), blocks, threads, 128);
        CudaKernels::ConstantPad2D(blocks, threads, output.Length(), input.GetDevicePtr(), input.Stride(1), input.Stride(2), input.Stride(3), left, right, top, bottom, value, output.GetDevicePtr(), output.Stride(1), output.Stride(2), output.Stride(3));
        cudaStreamSynchronize(0);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::ReflectPad2D(const Tensor& input, uint32_t left, uint32_t right, uint32_t top, uint32_t bottom, Tensor& output) const
    {
        NVTXProfile nvtxProfile(__FUNCTION__, 0xFF004A7F);
        input.CopyToDevice();
        output.OverrideDevice();

        dim3 blocks, threads;
        GetKernelRunParamsForSequence(output.Length(), blocks, threads, 128);
        CudaKernels::ReflectPad2D(blocks, threads, output.Length(), input.GetDevicePtr(), input.Stride(1), input.Stride(2), input.Stride(3), left, right, top, bottom, output.GetDevicePtr(), output.Stride(1), output.Stride(2), output.Stride(3));
        cudaStreamSynchronize(0);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::Pad2DGradient(const Tensor& gradient, uint32_t left, uint32_t right, uint32_t top, uint32_t bottom, Tensor& inputsGradient) const
    {
        NVTXProfile nvtxProfile(__FUNCTION__, 0xFF004A7F);
        gradient.CopyToDevice();
        inputsGradient.OverrideDevice();

        dim3 blocks, threads;
        GetKernelRunParamsForSequence(gradient.Length(), blocks, threads, 128);
        CudaKernels::Pad2DGradient(blocks, threads, inputsGradient.Length(), gradient.GetDevicePtr(), gradient.Stride(1), gradient.Stride(2), gradient.Stride(3), left, right, top, bottom, inputsGradient.GetDevicePtr(), inputsGradient.Stride(1), inputsGradient.Stride(2), inputsGradient.Stride(3));
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
        cudnnFilterDescriptor_t kernelsDesc; cudnnCreateFilterDescriptor(&kernelsDesc);

        cudnnSetConvolution2dDescriptor(convolutionDesc, paddingY, paddingX, stride, stride, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
        cudnnSetFilter4dDescriptor(kernelsDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, kernels.GetShape().Dimensions[3], kernels.GetShape().Dimensions[2], kernels.GetShape().Dimensions[1], kernels.GetShape().Dimensions[0]);

        cudnnConvolutionFwdAlgo_t algo;
        CUDA_CHECK(cudnnGetConvolutionForwardAlgorithm(s_CudnnHandle, input.DeviceDesc(), kernelsDesc, convolutionDesc, output.DeviceDesc(), CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo));

        size_t workspaceSize;
        CUDA_CHECK(cudnnGetConvolutionForwardWorkspaceSize(s_CudnnHandle, input.DeviceDesc(), kernelsDesc, convolutionDesc, output.DeviceDesc(), algo, &workspaceSize));
        void* workspacePtr;
        DeviceMemoryManager::Default().Allocate(&workspacePtr, workspaceSize, "conv2d_workspace");

        float alpha = 1, beta = 0;
        CUDA_CHECK(cudnnConvolutionForward(
            s_CudnnHandle,
            &alpha,
            input.DeviceDesc(),
            input.GetDevicePtr(),
            kernelsDesc,
            kernels.GetDevicePtr(),
            convolutionDesc,
            algo,
            workspacePtr,
            workspaceSize,
            &beta,
            output.DeviceDesc(),
            output.GetDevicePtr()));

        CUDA_CHECK(cudaLaunchHostFunc(0, DeallocateWorkspace, workspacePtr));
        cudaStreamSynchronize(0);

        cudnnDestroyFilterDescriptor(kernelsDesc);
        cudnnDestroyConvolutionDescriptor(convolutionDesc);
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
        cudnnFilterDescriptor_t kernelsDesc; cudnnCreateFilterDescriptor(&kernelsDesc);

        cudnnSetConvolution2dDescriptor(convolutionDesc, paddingY, paddingX, stride, stride, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
        cudnnSetFilter4dDescriptor(kernelsDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, kernels.GetShape().Dimensions[3], kernels.GetShape().Dimensions[2], kernels.GetShape().Dimensions[1], kernels.GetShape().Dimensions[0]);

        cudnnConvolutionFwdAlgo_t algo;
        CUDA_CHECK(cudnnGetConvolutionForwardAlgorithm(s_CudnnHandle, input.DeviceDesc(), kernelsDesc, convolutionDesc, output.DeviceDesc(), CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo));

        cudnnSetActivationDescriptor(activationDesc, GetCudnnActivationMode(activation), CUDNN_NOT_PROPAGATE_NAN, activationAlpha);

        size_t workspaceSize;
        CUDA_CHECK(cudnnGetConvolutionForwardWorkspaceSize(s_CudnnHandle, input.DeviceDesc(), kernelsDesc, convolutionDesc, output.DeviceDesc(), algo, &workspaceSize));
        void* workspacePtr;
        DeviceMemoryManager::Default().Allocate(&workspacePtr, workspaceSize, "conv2d_workspace");

        float alpha = 1, beta = 0;
        CUDA_CHECK(cudnnConvolutionBiasActivationForward(
            s_CudnnHandle,
            &alpha,
            input.DeviceDesc(),
            input.GetDevicePtr(),
            kernelsDesc,
            kernels.GetDevicePtr(),
            convolutionDesc,
            algo,
            workspacePtr,
            workspaceSize,
            &beta,
            output.DeviceDesc(),
            output.GetDevicePtr(),
            bias.DeviceDesc(),
            bias.GetDevicePtr(),
            activationDesc,
            output.DeviceDesc(),
            output.GetDevicePtr()));

        CUDA_CHECK(cudaLaunchHostFunc(0, DeallocateWorkspace, workspacePtr));
        cudaStreamSynchronize(0);

        cudnnDestroyFilterDescriptor(kernelsDesc);
        cudnnDestroyActivationDescriptor(activationDesc);
        cudnnDestroyConvolutionDescriptor(convolutionDesc);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::Conv2DBiasGradient(const Tensor& gradient, Tensor& biasGradient)
    {
        NVTXProfile nvtxProfile(__FUNCTION__, 0xFF004A7F);
        gradient.CopyToDevice();
        biasGradient.OverrideDevice();
        biasGradient.Zero();

        float alpha = 1, beta = 0;
        CUDA_CHECK(cudnnConvolutionBackwardBias(
            s_CudnnHandle,
            &alpha,
            gradient.DeviceDesc(),
            gradient.GetDevicePtr(),
            &beta,
            biasGradient.DeviceDesc(),
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
        cudnnFilterDescriptor_t kernelsDesc; cudnnCreateFilterDescriptor(&kernelsDesc);
        
        cudnnSetConvolution2dDescriptor(convolutionDesc, paddingY, paddingX, stride, stride, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
        cudnnSetFilter4dDescriptor(kernelsDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, kernels.GetShape().Dimensions[3], kernels.GetShape().Dimensions[2], kernels.GetShape().Dimensions[1], kernels.GetShape().Dimensions[0]);

        cudnnConvolutionBwdDataAlgo_t algo;
        cudnnGetConvolutionBackwardDataAlgorithm(s_CudnnHandle, kernelsDesc, gradient.DeviceDesc(), convolutionDesc, inputGradient.DeviceDesc(), CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, &algo);

        size_t workspaceSize;
        cudnnGetConvolutionBackwardDataWorkspaceSize(s_CudnnHandle, kernelsDesc, gradient.DeviceDesc(), convolutionDesc, inputGradient.DeviceDesc(), algo, &workspaceSize);
        //inputGradient.m_GpuData.UpdateWorkspace(inputGradient.m_GpuData.m_ConvBackWorkspace, workspaceSize);
        void* workspacePtr;
        DeviceMemoryManager::Default().Allocate(&workspacePtr, workspaceSize, "conv2d_input_grad_workspace");

        float alpha = 1, beta = 0;
        CUDA_CHECK(cudnnConvolutionBackwardData(
            s_CudnnHandle,
            &alpha,
            kernelsDesc,
            kernels.GetDevicePtr(),
            gradient.DeviceDesc(),
            gradient.GetDevicePtr(),
            convolutionDesc,
            algo,
            workspacePtr,
            workspaceSize,
            &beta,
            inputGradient.DeviceDesc(),
            inputGradient.GetDevicePtr()));

        CUDA_CHECK(cudaLaunchHostFunc(0, DeallocateWorkspace, workspacePtr));
        cudaStreamSynchronize(0);

        cudnnDestroyFilterDescriptor(kernelsDesc);
        cudnnDestroyConvolutionDescriptor(convolutionDesc);
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
        cudnnFilterDescriptor_t kernelsGradientsDesc; cudnnCreateFilterDescriptor(&kernelsGradientsDesc);

        cudnnSetConvolution2dDescriptor(convolutionDesc, paddingY, paddingX, stride, stride, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
        cudnnSetFilter4dDescriptor(kernelsGradientsDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, kernelsGradient.GetShape().Dimensions[3], kernelsGradient.GetShape().Dimensions[2], kernelsGradient.GetShape().Dimensions[1], kernelsGradient.GetShape().Dimensions[0]);

        cudnnConvolutionBwdFilterAlgo_t algo;
        cudnnGetConvolutionBackwardFilterAlgorithm(s_CudnnHandle, input.DeviceDesc(), gradient.DeviceDesc(), convolutionDesc, kernelsGradientsDesc, CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &algo);

        size_t workspaceSize;
        cudnnGetConvolutionBackwardFilterWorkspaceSize(s_CudnnHandle, input.DeviceDesc(), gradient.DeviceDesc(), convolutionDesc, kernelsGradientsDesc, algo, &workspaceSize);
        void* workspacePtr;
        DeviceMemoryManager::Default().Allocate(&workspacePtr, workspaceSize, "conv2d_kernels_grad_workspace");

        float alpha = 1, beta = 0;
        CUDA_CHECK(cudnnConvolutionBackwardFilter(
            s_CudnnHandle,
            &alpha,
            input.DeviceDesc(),
            input.GetDevicePtr(),
            gradient.DeviceDesc(),
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

        cudnnDestroyFilterDescriptor(kernelsGradientsDesc);
        cudnnDestroyConvolutionDescriptor(convolutionDesc);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::Pool2D(const Tensor& input, uint32_t filterSize, uint32_t stride, EPoolingMode type, uint32_t paddingX, uint32_t paddingY, EDataFormat dataFormat, Tensor& output) const
    {
        NVTXProfile nvtxProfile(__FUNCTION__, 0xFF004A7F);
        input.CopyToDevice();
        output.OverrideDevice();

        cudnnPoolingDescriptor_t poolingDesc; cudnnCreatePoolingDescriptor(&poolingDesc);
        cudnnSetPooling2dDescriptor(poolingDesc, GetCudnnPoolType(type), CUDNN_NOT_PROPAGATE_NAN, filterSize, filterSize, paddingX, paddingY, stride, stride);

        float alpha = 1, beta = 0;
        CUDA_CHECK(cudnnPoolingForward(
            s_CudnnHandle,
            poolingDesc,
            &alpha,
            input.DeviceDesc(),
            input.GetDevicePtr(),
            &beta,
            output.DeviceDesc(),
            output.GetDevicePtr()));
        cudaStreamSynchronize(0);

        cudnnDestroyPoolingDescriptor(poolingDesc);
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
        cudnnSetPooling2dDescriptor(poolingDesc, GetCudnnPoolType(type), CUDNN_NOT_PROPAGATE_NAN, filterSize, filterSize, paddingX, paddingY, stride, stride);

        float alpha = 1, beta = 0;
        CUDA_CHECK(cudnnPoolingBackward(
            s_CudnnHandle,
            poolingDesc,
            &alpha,
            output.DeviceDesc(),
            output.GetDevicePtr(),
            outputGradient.DeviceDesc(),
            outputGradient.GetDevicePtr(),
            input.DeviceDesc(),
            input.GetDevicePtr(),
            &beta,
            inputGradient.DeviceDesc(),
            inputGradient.GetDevicePtr()));
        cudaStreamSynchronize(0);

        cudnnDestroyPoolingDescriptor(poolingDesc);
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

        float alpha = 1, _beta = 0;
        CUDA_CHECK(cudnnBatchNormalizationForwardInference(
            s_CudnnHandle,
            GetCudnnBatchNormMode(mode),
            &alpha,
            &_beta,
            input.DeviceDesc(),
            input.GetDevicePtr(),
            output.DeviceDesc(),
            output.GetDevicePtr(),
            gamma.DeviceDesc(),
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
        const auto& inputShape = input.GetShape();

        if (mode == Instance)
            return __super::BatchNormalizationTrain(input, mode, gamma, beta, momentum, epsilon, runningMean, runningVar, saveMean, saveInvVariance, output);
        if (mode == Spatial && (inputShape.Width() * inputShape.Height() * inputShape.Batch()) == 1) //edge case is handled gracefully in hand-made implementation
            return __super::BatchNormalizationTrain(input, mode, gamma, beta, momentum, epsilon, runningMean, runningVar, saveMean, saveInvVariance, output);
        if (mode == PerActivation && inputShape.Batch() == 1) //edge case is handled gracefully in hand-made implementation
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

        float alpha = 1, _beta = 0;
        CUDA_CHECK(cudnnBatchNormalizationForwardTraining(
            s_CudnnHandle,
            GetCudnnBatchNormMode(mode),
            &alpha,
            &_beta,
            input.DeviceDesc(),
            input.GetDevicePtr(),
            output.DeviceDesc(),
            output.GetDevicePtr(),
            gamma.DeviceDesc(),
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
        const auto& inputShape = input.GetShape();

        if (mode == Instance)
            return __super::BatchNormalizationGradient(input, mode, gamma, epsilon, outputGradient, savedMean, savedInvVariance, gammaGradient, betaGradient, trainable, inputGradient);
        if (mode == Spatial && (inputShape.Width() * inputShape.Height() * inputShape.Batch()) == 1) //edge case is handled gracefully in hand-made implementation
            return __super::BatchNormalizationGradient(input, mode, gamma, epsilon, outputGradient, savedMean, savedInvVariance, gammaGradient, betaGradient, trainable, inputGradient);
        if (mode == PerActivation && inputShape.Batch() == 1) //edge case is handled gracefully in hand-made implementation
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

        float alpha = 1.f, beta = 0.f, paramsGradAlpha = (trainable ? 1.f : 0.f), paramsGradBeta = 0.f;
        CUDA_CHECK(cudnnBatchNormalizationBackward(
            s_CudnnHandle,
            GetCudnnBatchNormMode(mode),
            &alpha,
            &beta,
            &paramsGradAlpha,
            &paramsGradBeta,
            input.DeviceDesc(),
            input.GetDevicePtr(),
            outputGradient.DeviceDesc(),
            outputGradient.GetDevicePtr(),
            inputGradient.DeviceDesc(),
            inputGradient.GetDevicePtr(),
            gamma.DeviceDesc(),
            gamma.GetDevicePtr(),
            gammaGradient.GetDevicePtr(),
            betaGradient.GetDevicePtr(),
            epsilon,
            savedMean.GetDevicePtr(),
            savedInvVariance.GetDevicePtr()));
        cudaStreamSynchronize(0);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::Dropout(const Tensor& input, float prob, Tensor& saveMask, Tensor& output) const
    {
        NVTXProfile nvtxProfile(__FUNCTION__, 0xFF004A7F);
        input.CopyToDevice();
        saveMask.OverrideDevice();
        output.OverrideDevice();

        NEURO_ASSERT(saveMask.Length() == input.Length(), "Mismatched mask and input length.");
        
        curandGenerateUniform(s_CurandGenerator, saveMask.GetDevicePtr(), saveMask.Length());        

        dim3 blocks, threads;
        GetKernelRunParamsForSequence(input.Length(), blocks, threads, 128);
        CudaKernels::Dropout(blocks, threads, input.Length(), input.GetDevicePtr(), prob, saveMask.GetDevicePtr(), output.GetDevicePtr());
        cudaStreamSynchronize(0);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::DropoutNoRand(const Tensor& input, float prob, Tensor& saveMask, Tensor& output) const
    {
        input.CopyToDevice();
        saveMask.CopyToDevice();
        output.OverrideDevice();

        dim3 blocks, threads;
        GetKernelRunParamsForSequence(input.Length(), blocks, threads, 128);
        CudaKernels::Dropout(blocks, threads, input.Length(), input.GetDevicePtr(), prob, saveMask.GetDevicePtr(), output.GetDevicePtr());
        cudaStreamSynchronize(0);
    }

    //////////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::DropoutGradient(const Tensor& outputGradient, float prob, const Tensor& savedMask, Tensor& inputGradient) const
    {
        NVTXProfile nvtxProfile(__FUNCTION__, 0xFF004A7F);
        outputGradient.CopyToDevice();
        savedMask.CopyToDevice();
        inputGradient.OverrideDevice();

        dim3 blocks, threads;
        GetKernelRunParamsForSequence(inputGradient.Length(), blocks, threads, 128);
        CudaKernels::DropoutGradient(blocks, threads, inputGradient.Length(), outputGradient.GetDevicePtr(), savedMask.GetDevicePtr(), inputGradient.GetDevicePtr());
        cudaStreamSynchronize(0);
    }

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
        input.CopyToDevice();
        output.OverrideDevice();

        dim3 blocks, threads;
        GetKernelRunParamsForSequence(input.Length(), blocks, threads, 128);
        CudaKernels::LeakyReLU(blocks, threads, input.Length(), input.GetDevicePtr(), alpha, output.GetDevicePtr());
        cudaStreamSynchronize(0);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::LeakyReLUGradient(const Tensor& output, const Tensor& outputGradient, float alpha, Tensor& inputGradient) const
    {
        NVTXProfile nvtxProfile(__FUNCTION__, 0xFF004A7F);
        output.CopyToDevice();
        outputGradient.CopyToDevice();
        inputGradient.OverrideDevice();
        
        dim3 blocks, threads;
        GetKernelRunParamsForSequence(output.Length(), blocks, threads, 128);
        CudaKernels::LeakyReLUGradient(blocks, threads, output.Length(), output.GetDevicePtr(), outputGradient.GetDevicePtr(), alpha, inputGradient.GetDevicePtr());
        cudaStreamSynchronize(0);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::Softmax(const Tensor& input, Tensor& output) const
    {
        NVTXProfile nvtxProfile(__FUNCTION__, 0xFF004A7F);
        input.CopyToDevice();
        output.OverrideDevice();

        float alpha = 1, beta = 0;
        CUDA_CHECK(cudnnSoftmaxForward(
            s_CudnnHandle,
            CUDNN_SOFTMAX_ACCURATE,
            CUDNN_SOFTMAX_MODE_INSTANCE,
            &alpha,
            input.DeviceDesc(),
            input.GetDevicePtr(),
            &beta,
            output.DeviceDesc(),
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

        float alpha = 1, beta = 0;
        CUDA_CHECK(cudnnSoftmaxBackward(
            s_CudnnHandle,
            CUDNN_SOFTMAX_ACCURATE,
            CUDNN_SOFTMAX_MODE_INSTANCE,
            &alpha,
            output.DeviceDesc(),
            output.GetDevicePtr(),
            outputGradient.DeviceDesc(),
            outputGradient.GetDevicePtr(),
            &beta,
            inputGradient.DeviceDesc(),
            inputGradient.GetDevicePtr()));
        cudaStreamSynchronize(0);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::ExtractSubTensor2D(const Tensor& input, uint32_t widthOffset, uint32_t heightOffset, Tensor& output) const
    {
        NVTXProfile nvtxProfile(__FUNCTION__, 0xFF004A7F);
        input.CopyToDevice();
        output.OverrideDevice();

        dim3 blocks, threads;
        GetKernelRunParamsForSequence(output.Length(), blocks, threads, 128);
        CudaKernels::ExtractSubTensor2D(blocks, threads, output.Length(), input.GetDevicePtr(), input.Stride(1), input.Stride(2), input.Stride(3), widthOffset, heightOffset, output.GetDevicePtr(), output.Stride(1), output.Stride(2), output.Stride(3));
        cudaStreamSynchronize(0);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::FuseSubTensor2D(const Tensor& input, uint32_t widthOffset, uint32_t heightOffset, Tensor& output) const
    {
        NVTXProfile nvtxProfile(__FUNCTION__, 0xFF004A7F);
        input.CopyToDevice();
        output.CopyToDevice();

        dim3 blocks, threads;
        GetKernelRunParamsForSequence(input.Length(), blocks, threads, 128);
        CudaKernels::FuseSubTensor2D(blocks, threads, input.Length(), input.GetDevicePtr(), input.Stride(1), input.Stride(2), input.Stride(3), widthOffset, heightOffset, output.GetDevicePtr(), output.Stride(1), output.Stride(2), output.Stride(3));
        cudaStreamSynchronize(0);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::Reduce(const Tensor& input, cudnnReduceTensorOp_t reductionOp, Tensor& output) const
    {
        NVTXProfile nvtxProfile(__FUNCTION__, 0xFF004A7F);
        input.CopyToDevice();
        output.OverrideDevice();

        cudnnReduceTensorDescriptor_t reduceDesc; cudnnCreateReduceTensorDescriptor(&reduceDesc);
        cudnnSetReduceTensorDescriptor(reduceDesc, reductionOp, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN, CUDNN_REDUCE_TENSOR_NO_INDICES, CUDNN_32BIT_INDICES);
        
        size_t workspaceSize;
        CUDA_CHECK(cudnnGetReductionWorkspaceSize(s_CudnnHandle, reduceDesc, input.DeviceDesc(), output.DeviceDesc(), &workspaceSize));
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
            input.DeviceDesc(),
            input.GetDevicePtr(),
            &beta,
            output.DeviceDesc(),
            output.GetDevicePtr()));

        CUDA_CHECK(cudaLaunchHostFunc(0, DeallocateWorkspace, workspacePtr));
        cudaStreamSynchronize(0);

        cudnnDestroyReduceTensorDescriptor(reduceDesc);
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
    void TensorOpGpu::AdamStep(Tensor& parameter, const Tensor& gradient, Tensor& mGrad, Tensor& vGrad, float lr, float beta1, float beta2, float epsilon) const
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
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::SgdStep(Tensor& parameter, const Tensor& gradient, float lr) const
    {
        NVTXProfile nvtxProfile(__FUNCTION__, 0xFF004A7F);
        parameter.CopyToDevice();
        gradient.CopyToDevice();

        dim3 blocks, threads;
        GetKernelRunParams(parameter.Length(), blocks, threads, s_CudaDevProp.maxThreadsPerBlock);

        CudaKernels::SgdStep(blocks, threads, parameter.Length(), parameter.GetDevicePtr(), gradient.GetDevicePtr(), lr);
        cudaStreamSynchronize(0);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::Activation(const cudnnActivationMode_t& activationMode, const Tensor& input, Tensor& output, float coeff) const
    {
        NVTXProfile nvtxProfile(__FUNCTION__, 0xFF004A7F);
        input.CopyToDevice();
        output.OverrideDevice();

        cudnnActivationDescriptor_t activationDesc; cudnnCreateActivationDescriptor(&activationDesc);
        cudnnSetActivationDescriptor(activationDesc, activationMode, CUDNN_PROPAGATE_NAN, coeff);

        float alpha = 1, beta = 0;
        CUDA_CHECK(cudnnActivationForward(
            s_CudnnHandle,
            activationDesc,
            &alpha,
            input.DeviceDesc(),
            input.GetDevicePtr(),
            &beta,
            output.DeviceDesc(),
            output.GetDevicePtr()));
        cudaStreamSynchronize(0);

        cudnnDestroyActivationDescriptor(activationDesc);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::ActivationGradient(const cudnnActivationMode_t& activationMode, const Tensor& output, const Tensor& outputGradient, Tensor& inputGradient, float coeff) const
    {
        NVTXProfile nvtxProfile(__FUNCTION__, 0xFF004A7F);
        output.CopyToDevice();
        outputGradient.CopyToDevice();
        inputGradient.OverrideDevice();
        inputGradient.Zero();

        cudnnActivationDescriptor_t activationDesc; cudnnCreateActivationDescriptor(&activationDesc);
        cudnnSetActivationDescriptor(activationDesc, activationMode, CUDNN_PROPAGATE_NAN, coeff);

        float alpha = 1, beta = 0;
        CUDA_CHECK(cudnnActivationBackward(
            s_CudnnHandle,
            activationDesc,
            &alpha,
            output.DeviceDesc(),
            output.GetDevicePtr(),
            outputGradient.DeviceDesc(),
            outputGradient.GetDevicePtr(),
            output.DeviceDesc(),
            output.GetDevicePtr(),
            &beta,
            inputGradient.DeviceDesc(),
            inputGradient.GetDevicePtr()));
        cudaStreamSynchronize(0);

        cudnnDestroyActivationDescriptor(activationDesc);
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
    void TensorOpGpu::GetKernelRunParamsForSequence(int count, dim3& blocks, dim3& threads, int maxThreads)
    {
        threads.x = maxThreads;
        blocks.x = (unsigned int)ceil((float)count / threads.x / s_CudaDevProp.multiProcessorCount);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::GetKernelRunParams(int count, dim3& blocks, dim3& threads, int maxThreads)
    {
        int blocksNum = (int)ceil(count / (float)maxThreads);

        if (count <= maxThreads)
        {
            blocksNum = 1;
            maxThreads = count;
        }

        blocks = dim3(blocksNum, 1, 1);
        threads = dim3(maxThreads, 1, 1);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::CudnnLog(cudnnSeverity_t sev, void *udata, const cudnnDebug_t *dbg, const char *msg)
    {
        OutputDebugString(msg);
        OutputDebugString("\n");
    }
}
