#include <sstream>
#include <windows.h>
#include <debugapi.h>

#include "Tools.h"
#include "Tensors/TensorOpGpu.h"
#include "Tensors/Cuda/CudaDeviceVariable.h"
#include "Tensors/Cuda/CudaErrorCheck.h"
#include "Tensors/Cuda/CudaKernels.h"

namespace Neuro
{
    bool TensorOpGpu::s_Initialized = false;
    cudaDeviceProp TensorOpGpu::s_CudaDevProp;
    cublasHandle_t TensorOpGpu::s_CublasHandle = nullptr;
    cudnnHandle_t TensorOpGpu::s_CudnnHandle = nullptr;

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
                stringstream ss;
                ss << "GPU: " << s_CudaDevProp.name << "(threads_per_block: " << s_CudaDevProp.maxThreadsPerBlock << ")\n";
                OutputDebugString(ss.str().c_str());
            }
        }
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::Add(float alpha, const Tensor& t1, float beta, const Tensor& t2, Tensor& output) const
    {
        t1.CopyToDevice();
        t2.CopyToDevice();
        output.CopyToDevice();

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

        for (uint32_t n = 0; n < t1.Batch(); ++n)
        {
            CUDA_CHECK(cublasSgeam(
                s_CublasHandle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                t1.BatchLength(),
                1,
                &alpha,
                CudaDeviceVariable<float>(t1.GetDeviceVar(), n * t1.BatchLength()).GetDevicePtr(),
                t1.BatchLength(),
                &beta,
                t2.GetDevicePtr(),
                t2.BatchLength(),
                CudaDeviceVariable<float>(output.GetDeviceVar(), n * output.BatchLength()).GetDevicePtr(),
                output.BatchLength()));
        }
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::Mul(bool transposeT1, bool transposeT2, const Tensor& t1, const Tensor& t2, Tensor& output) const
    {
        output.Zero();
        t1.CopyToDevice();
        t2.CopyToDevice();
        output.CopyToDevice();

        if (t1.Depth() == t2.Depth() && t1.Batch() == t2.Batch())
            MulStridedBatched(transposeT1, transposeT2, t1, t2, output);
        else if (t1.Depth() * output.Batch() > 48)
            MulBatched(transposeT1, transposeT2, t1, t2, output);
        else
            MulGeneric(transposeT1, transposeT2, t1, t2, output);
    }

    void TensorOpGpu::Div(const Tensor& input, float v, Tensor& output) const
    {
        dim3 blocks, threads;
        GetKernelRunParams(input.Length(), blocks, threads);
        input.CopyToDevice();
        output.CopyToDevice();

        CudaKernels::Div(blocks, threads, input.Length(), input.GetDevicePtr(), v, output.GetDevicePtr());
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::Transpose(const Tensor& input, Tensor& output) const
    {
        input.CopyToDevice();
        output.CopyToDevice();

        int m = input.Height();
        uint32_t n = input.Width();

        //treat depth as batch
        int batches = input.Depth() * input.Batch();
        float alpha = 1, beta = 0;

        for (int b = 0; b < batches; ++b)
        {
            CudaDeviceVariable<float> tVar(input.GetDeviceVar(), b * input.GetShape().Dim0Dim1);

            CUDA_CHECK(cublasSgeam(
                s_CublasHandle,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                m,
                n,  // trick to convert row major to column major
                &alpha,
                tVar.GetDevicePtr(),
                n,
                &beta,
                tVar.GetDevicePtr(),
                m,
                CudaDeviceVariable<float>(output.GetDeviceVar(), b * output.GetShape().Dim0Dim1).GetDevicePtr(), m));
        }
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::Conv2D(const Tensor& input, const Tensor& kernels, uint32_t stride, uint32_t paddingX, uint32_t paddingY, Tensor& output) const
    {
        output.Zero();
        input.CopyToDevice();
        kernels.CopyToDevice();
        output.CopyToDevice();

        cudnnConvolutionDescriptor_t convolutionDesc; cudnnCreateConvolutionDescriptor(&convolutionDesc);
        cudnnTensorDescriptor_t inputDesc; cudnnCreateTensorDescriptor(&inputDesc);
        cudnnFilterDescriptor_t kernelsDesc; cudnnCreateFilterDescriptor(&kernelsDesc);
        cudnnTensorDescriptor_t outputDesc; cudnnCreateTensorDescriptor(&outputDesc);

        cudnnSetConvolution2dDescriptor(convolutionDesc, paddingY, paddingX, stride, stride, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
        cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, input.GetShape().Dimensions[3], input.GetShape().Dimensions[2], input.GetShape().Dimensions[1], input.GetShape().Dimensions[0]);
        cudnnSetFilter4dDescriptor(kernelsDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, kernels.GetShape().Dimensions[3], kernels.GetShape().Dimensions[2], kernels.GetShape().Dimensions[1], kernels.GetShape().Dimensions[0]);
        cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, output.GetShape().Dimensions[3], output.GetShape().Dimensions[2], output.GetShape().Dimensions[1], output.GetShape().Dimensions[0]);

        cudnnConvolutionFwdAlgo_t algo;
        CUDA_CHECK(cudnnGetConvolutionForwardAlgorithm(s_CudnnHandle, inputDesc, kernelsDesc, convolutionDesc, outputDesc, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo));

        size_t workspaceSize;
        CUDA_CHECK(cudnnGetConvolutionForwardWorkspaceSize(s_CudnnHandle, inputDesc, kernelsDesc, convolutionDesc, outputDesc, algo, &workspaceSize));
        output.m_GpuData.UpdateWorkspace(output.m_GpuData.m_ConvWorkspace, workspaceSize);

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
            output.m_GpuData.m_ConvWorkspace->GetDevicePtr(),
            workspaceSize,
            &beta,
            outputDesc,
            output.GetDevicePtr()));
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::Conv2DInputGradient(const Tensor& gradient, const Tensor& kernels, uint32_t stride, uint32_t paddingX, uint32_t paddingY, Tensor& inputGradient) const
    {
        gradient.CopyToDevice();
        kernels.CopyToDevice();
        inputGradient.CopyToDevice();

        cudnnConvolutionDescriptor_t convolutionDesc; cudnnCreateConvolutionDescriptor(&convolutionDesc);
        cudnnTensorDescriptor_t gradientDesc; cudnnCreateTensorDescriptor(&gradientDesc);
        cudnnFilterDescriptor_t kernelsDesc; cudnnCreateFilterDescriptor(&kernelsDesc);
        cudnnTensorDescriptor_t inputGradientDesc; cudnnCreateTensorDescriptor(&inputGradientDesc);

        cudnnSetConvolution2dDescriptor(convolutionDesc, paddingY, paddingX, stride, stride, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
        cudnnSetTensor4dDescriptor(gradientDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, gradient.GetShape().Dimensions[3], gradient.GetShape().Dimensions[2], gradient.GetShape().Dimensions[1], gradient.GetShape().Dimensions[0]);
        cudnnSetFilter4dDescriptor(kernelsDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, kernels.GetShape().Dimensions[3], kernels.GetShape().Dimensions[2], kernels.GetShape().Dimensions[1], kernels.GetShape().Dimensions[0]);
        cudnnSetTensor4dDescriptor(inputGradientDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, inputGradient.GetShape().Dimensions[3], inputGradient.GetShape().Dimensions[2], inputGradient.GetShape().Dimensions[1], inputGradient.GetShape().Dimensions[0]);

        cudnnConvolutionBwdDataAlgo_t algo;
        cudnnGetConvolutionBackwardDataAlgorithm(s_CudnnHandle, kernelsDesc, gradientDesc, convolutionDesc, inputGradientDesc, CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, &algo);

        size_t workspaceSize;
        cudnnGetConvolutionBackwardDataWorkspaceSize(s_CudnnHandle, kernelsDesc, gradientDesc, convolutionDesc, inputGradientDesc, algo, &workspaceSize);
        inputGradient.m_GpuData.UpdateWorkspace(inputGradient.m_GpuData.m_ConvBackWorkspace, workspaceSize);

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
            inputGradient.m_GpuData.m_ConvBackWorkspace->GetDevicePtr(),
            workspaceSize,
            &beta,
            inputGradientDesc,
            inputGradient.GetDevicePtr()));
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::Conv2DKernelsGradient(const Tensor& input, const Tensor& gradient, uint32_t stride, uint32_t paddingX, uint32_t paddingY, Tensor& kernelsGradient) const
    {
        gradient.CopyToDevice();
        input.CopyToDevice();
        kernelsGradient.CopyToDevice();

        cudnnConvolutionDescriptor_t convolutionDesc; cudnnCreateConvolutionDescriptor(&convolutionDesc);
        cudnnTensorDescriptor_t gradientDesc; cudnnCreateTensorDescriptor(&gradientDesc);
        cudnnTensorDescriptor_t inputDesc; cudnnCreateTensorDescriptor(&inputDesc);
        cudnnFilterDescriptor_t kernelsGradientsDesc; cudnnCreateFilterDescriptor(&kernelsGradientsDesc);

        cudnnSetConvolution2dDescriptor(convolutionDesc, paddingY, paddingX, stride, stride, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
        cudnnSetTensor4dDescriptor(gradientDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, gradient.GetShape().Dimensions[3], gradient.GetShape().Dimensions[2], gradient.GetShape().Dimensions[1], gradient.GetShape().Dimensions[0]);
        cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, input.GetShape().Dimensions[3], input.GetShape().Dimensions[2], input.GetShape().Dimensions[1], input.GetShape().Dimensions[0]);
        cudnnSetFilter4dDescriptor(kernelsGradientsDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, kernelsGradient.GetShape().Dimensions[3], kernelsGradient.GetShape().Dimensions[2], kernelsGradient.GetShape().Dimensions[1], kernelsGradient.GetShape().Dimensions[0]);

        cudnnConvolutionBwdFilterAlgo_t algo;
        cudnnGetConvolutionBackwardFilterAlgorithm(s_CudnnHandle, inputDesc, gradientDesc, convolutionDesc, kernelsGradientsDesc, CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &algo);

        size_t workspaceSize;
        cudnnGetConvolutionBackwardFilterWorkspaceSize(s_CudnnHandle, inputDesc, gradientDesc, convolutionDesc, kernelsGradientsDesc, algo, &workspaceSize);
        kernelsGradient.m_GpuData.UpdateWorkspace(kernelsGradient.m_GpuData.m_ConvBackKernelWorkspace, workspaceSize);

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
            kernelsGradient.m_GpuData.m_ConvBackKernelWorkspace->GetDevicePtr(),
            workspaceSize,
            &beta,
            kernelsGradientsDesc,
            kernelsGradient.GetDevicePtr()));
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::Pool2D(const Tensor& input, uint32_t filterSize, uint32_t stride, EPoolingMode type, uint32_t paddingX, uint32_t paddingY, Tensor& output) const
    {
        input.CopyToDevice();
        output.CopyToDevice();

        cudnnPoolingDescriptor_t poolingDesc; cudnnCreatePoolingDescriptor(&poolingDesc);
        cudnnTensorDescriptor_t inputDesc; cudnnCreateTensorDescriptor(&inputDesc);
        cudnnTensorDescriptor_t outputDesc; cudnnCreateTensorDescriptor(&outputDesc);

        cudnnSetPooling2dDescriptor(poolingDesc, GetCudnnPoolType(type), CUDNN_NOT_PROPAGATE_NAN, filterSize, filterSize, paddingX, paddingY, stride, stride);
        cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, input.GetShape().Dimensions[3], input.GetShape().Dimensions[2], input.GetShape().Dimensions[1], input.GetShape().Dimensions[0]);
        cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, output.GetShape().Dimensions[3], output.GetShape().Dimensions[2], output.GetShape().Dimensions[1], output.GetShape().Dimensions[0]);

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
    void TensorOpGpu::Pool2DGradient(const Tensor& output, const Tensor& input, const Tensor& outputGradient, uint32_t filterSize, uint32_t stride, EPoolingMode type, uint32_t paddingX, uint32_t paddingY, Tensor& inputGradient) const
    {
        output.CopyToDevice();
        input.CopyToDevice();
        outputGradient.CopyToDevice();
        inputGradient.CopyToDevice();

        cudnnPoolingDescriptor_t poolingDesc; cudnnCreatePoolingDescriptor(&poolingDesc);
        cudnnTensorDescriptor_t outputDesc; cudnnCreateTensorDescriptor(&outputDesc);
        cudnnTensorDescriptor_t inputDesc; cudnnCreateTensorDescriptor(&inputDesc);
        cudnnTensorDescriptor_t outputGradientDesc; cudnnCreateTensorDescriptor(&outputGradientDesc);
        cudnnTensorDescriptor_t inputGradientDesc; cudnnCreateTensorDescriptor(&inputGradientDesc);

        cudnnSetPooling2dDescriptor(poolingDesc, GetCudnnPoolType(type), CUDNN_NOT_PROPAGATE_NAN, filterSize, filterSize, paddingX, paddingY, stride, stride);
        cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, output.GetShape().Dimensions[3], output.GetShape().Dimensions[2], output.GetShape().Dimensions[1], output.GetShape().Dimensions[0]);
        cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, input.GetShape().Dimensions[3], input.GetShape().Dimensions[2], input.GetShape().Dimensions[1], input.GetShape().Dimensions[0]);
        cudnnSetTensor4dDescriptor(outputGradientDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, outputGradient.GetShape().Dimensions[3], outputGradient.GetShape().Dimensions[2], outputGradient.GetShape().Dimensions[1], outputGradient.GetShape().Dimensions[0]);
        cudnnSetTensor4dDescriptor(inputGradientDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, inputGradient.GetShape().Dimensions[3], inputGradient.GetShape().Dimensions[2], inputGradient.GetShape().Dimensions[1], inputGradient.GetShape().Dimensions[0]);

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

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::BatchNormalization(const Tensor& input, EBatchNormMode mode, const Tensor& gamma, const Tensor& beta, float epsilon, const Tensor& runningMean, const Tensor& runningVar, Tensor& output) const
    {
        input.CopyToDevice();
        gamma.CopyToDevice();
        beta.CopyToDevice();
        runningMean.CopyToDevice();
        runningVar.CopyToDevice();
        output.CopyToDevice();

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
            runningMean.GetDevicePtr(),
            runningVar.GetDevicePtr(),
            epsilon));
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::BatchNormalizationTrain(const Tensor& input, EBatchNormMode mode, const Tensor& gamma, const Tensor& beta, float momentum, float epsilon, Tensor& runningMean, Tensor& runningVar, Tensor& saveMean, Tensor& saveInvVariance, Tensor& output) const
    {
        input.CopyToDevice();
        gamma.CopyToDevice();
        beta.CopyToDevice();
        runningMean.CopyToDevice();
        runningVar.CopyToDevice();
        saveMean.CopyToDevice();
        saveInvVariance.CopyToDevice();
        output.CopyToDevice();

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
            runningMean.GetDevicePtr(),
            runningVar.GetDevicePtr(),
            epsilon,
            saveMean.GetDevicePtr(),
            saveInvVariance.GetDevicePtr()));
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::BatchNormalizationGradient(const Tensor& input, EBatchNormMode mode, const Tensor& gamma, float epsilon, const Tensor& outputGradient, const Tensor& savedMean, const Tensor& savedInvVariance, Tensor& gammaGradient, Tensor& betaGradient, bool trainable, Tensor& inputGradient) const
    {
        input.CopyToDevice();
        gamma.CopyToDevice();
        outputGradient.CopyToDevice();
        savedMean.CopyToDevice();
        savedInvVariance.CopyToDevice();
        gammaGradient.CopyToDevice();
        betaGradient.CopyToDevice();
        inputGradient.CopyToDevice();

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
    //void TensorOpGpu::Dropout(const Tensor& input, float prob, Tensor& saveMask, Tensor& output)
    //{
    //    input.CopyToDevice();
    //    output.CopyToDevice();
    //    saveMask.FillWithFunc([&]() { return GlobalRng().NextFloat() < prob ? 0.f : 1.f; });
    //    saveMask.CopyToDevice();

    //    cudnnTensorDescriptor_t inputOutputDesc; cudnnCreateTensorDescriptor(&inputOutputDesc);
    //    cudnnDropoutDescriptor_t dropoutDesc; cudnnCreateDropoutDescriptor(&dropoutDesc);

    //    cudnnSetTensor4dDescriptor(inputOutputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, input.GetShape().Dimensions[3], input.GetShape().Dimensions[2], input.GetShape().Dimensions[1], input.GetShape().Dimensions[0]);
    //    cudnnSetDropoutDescriptor(dropoutDesc, s_CudnnHandle, prob, saveMask.GetDevicePtr(), saveMask.Length() * sizeof(float), 0);

    //    size_t dropoutReserveSize;
    //    CUDA_CHECK(cudnnDropoutGetReserveSpaceSize(inputOutputDesc, &dropoutReserveSize));
    //    output.m_GpuData.UpdateWorkspace(output.m_GpuData.m_DropoutWorkspace, dropoutReserveSize);

    //    CUDA_CHECK(cudnnDropoutForward(
    //        s_CudnnHandle,
    //        dropoutDesc,
    //        inputOutputDesc,
    //        input.GetDevicePtr(),
    //        inputOutputDesc,
    //        output.GetDevicePtr(),
    //        output.m_GpuData.m_DropoutWorkspace->GetDevicePtr(),
    //        dropoutReserveSize));
    //}

    ////////////////////////////////////////////////////////////////////////////
    //void TensorOpGpu::DropoutGradient(const Tensor& outputGradient, const Tensor& savedMask, Tensor& inputGradient)
    //{

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
        GetKernelRunParams(input.Length(), blocks, threads);
        input.CopyToDevice();
        output.CopyToDevice();

        CudaKernels::LeakyReLU(blocks, threads, input.Length(), input.GetDevicePtr(), alpha, output.GetDevicePtr());
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::LeakyReLUGradient(const Tensor& output, const Tensor& outputGradient, float alpha, Tensor& inputGradient) const
    {
        dim3 blocks, threads;
        GetKernelRunParams(output.Length(), blocks, threads);
        output.CopyToDevice();
        outputGradient.CopyToDevice();
        inputGradient.CopyToDevice();

        CudaKernels::LeakyReLUGradient(blocks, threads, output.Length(), output.GetDevicePtr(), outputGradient.GetDevicePtr(), alpha, inputGradient.GetDevicePtr());
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::Softmax(const Tensor& input, Tensor& output) const
    {
        input.CopyToDevice();
        output.CopyToDevice();

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
        inputGradient.CopyToDevice();

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
        if (axis != EAxis::BatchAxis)
            return __super::Sum(input, axis, output);

        output.Zero();
        input.CopyToDevice();
        output.CopyToDevice();

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
                CudaDeviceVariable<float>(input.GetDeviceVar(), n * input.BatchLength()).GetDevicePtr(),
                input.BatchLength(),
                &beta,
                output.GetDevicePtr(),
                output.BatchLength(),
                output.GetDevicePtr(),
                output.BatchLength()));
        }
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::AdamStep(Tensor& parameter, Tensor& gradient, Tensor& mGrad, Tensor& vGrad, float batchSize, float lr, float beta1, float beta2, float epsilon) const
    {
        parameter.CopyToDevice();
        gradient.CopyToDevice();
        mGrad.CopyToDevice();
        vGrad.CopyToDevice();

        dim3 blocks, threads;
        GetKernelRunParams(parameter.Length(), blocks, threads);
        
        CudaKernels::AdamStep(blocks, threads, parameter.Length(), parameter.GetDevicePtr(), gradient.GetDevicePtr(), mGrad.GetDevicePtr(), vGrad.GetDevicePtr(), batchSize, lr, beta1, beta2, epsilon);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::SgdStep(Tensor& parameter, Tensor& gradient, float batchSize, float lr) const
    {
        parameter.CopyToDevice();
        gradient.CopyToDevice();

        dim3 blocks, threads;
        GetKernelRunParams(parameter.Length(), blocks, threads);

        CudaKernels::SgdStep(blocks, threads, parameter.Length(), parameter.GetDevicePtr(), gradient.GetDevicePtr(), batchSize, lr);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::Activation(const cudnnActivationMode_t& activationMode, const Tensor& input, Tensor& output, float coeff) const
    {
        input.CopyToDevice();
        output.CopyToDevice();

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
        inputGradient.CopyToDevice();

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
                    CudaDeviceVariable<float>(t2.GetDeviceVar(), d * t2.GetShape().Dim0Dim1 + t2B * t2.BatchLength()).GetDevicePtr(),
                    n,
                    CudaDeviceVariable<float>(t1.GetDeviceVar(), d * t1.GetShape().Dim0Dim1 + t1B * t1.BatchLength()).GetDevicePtr(),
                    k,
                    &beta,
                    CudaDeviceVariable<float>(output.GetDeviceVar(), d * output.GetShape().Dim0Dim1 + b * output.BatchLength()).GetDevicePtr(),
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
        vector<float*> t1List(batches);
        vector<float*> t2List(batches);
        vector<float*> outputList(batches);

        for (uint32_t b = 0; b < output.Batch(); ++b)
        {
            uint32_t t1B = min(b, t1.Batch() - 1);
            uint32_t t2B = min(b, t2.Batch() - 1);

            for (uint32_t d = 0; d < t1.Depth(); ++d)
            {
                uint32_t idx = b * t1.Depth() + d;
                t1List[idx] = CudaDeviceVariable<float>(t1.GetDeviceVar(), d * t1.GetShape().Dim0Dim1 + t1B * t1.BatchLength()).GetDevicePtr();
                t2List[idx] = CudaDeviceVariable<float>(t2.GetDeviceVar(), d * t2.GetShape().Dim0Dim1 + t2B * t2.BatchLength()).GetDevicePtr();
                outputList[idx] = CudaDeviceVariable<float>(output.GetDeviceVar(), d * output.GetShape().Dim0Dim1 + b * output.BatchLength()).GetDevicePtr();
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
        if (mode == EPoolingMode::Max)
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
    void TensorOpGpu::GetKernelRunParams(int count, dim3& blocks, dim3& threads)
    {
        int threadsPerBlock = s_CudaDevProp.maxThreadsPerBlock;
        int blockCount = GetBlocksNum(count);

        if (count <= s_CudaDevProp.maxThreadsPerBlock)
        {
            blockCount = 1;
            threadsPerBlock = count;
        }

        blocks = dim3(blockCount, 1, 1);
        threads = dim3(threadsPerBlock, 1, 1);
    }

    //////////////////////////////////////////////////////////////////////////
    int TensorOpGpu::GetBlocksNum(int count)
    {
        return (int)ceil(count / (float)s_CudaDevProp.maxThreadsPerBlock);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::CudnnLog(cudnnSeverity_t sev, void *udata, const cudnnDebug_t *dbg, const char *msg)
    {
        OutputDebugString(msg);
        OutputDebugString("\n");
    }
}
