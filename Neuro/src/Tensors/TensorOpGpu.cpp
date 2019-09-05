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
                ss << "GPU: " << s_CudaDevProp.name << "(tpb: " << s_CudaDevProp.maxThreadsPerBlock << ")\n";
                OutputDebugString(ss.str().c_str());
            }
        }
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::Add(float alpha, const Tensor& t1, float beta, const Tensor& t2, Tensor& result) const
    {
        t1.CopyToDevice();
        t2.CopyToDevice();
        result.CopyToDevice();

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
                result.GetDevicePtr(), 
                result.Length()));
            return;
        }

        for (int n = 0; n < t1.Batch(); ++n)
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
                CudaDeviceVariable<float>(result.GetDeviceVar(), n * result.BatchLength()).GetDevicePtr(), 
                result.BatchLength()));
        }
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::Mul(bool transposeT1, bool transposeT2, const Tensor& t1, const Tensor& t2, Tensor& result) const
    {
        result.Zero();
        t1.CopyToDevice();
        t2.CopyToDevice();
        result.CopyToDevice();

        int m = t1.Height();
        int n = t2.Width();
        int k = t1.Width();

        float alpha = 1, beta = 0;

        for (int b = 0; b < result.Batch(); ++b)
        {
            int t1B = min(b, t1.Batch() - 1);
            int t2B = min(b, t2.Batch() - 1);

            for (int d = 0; d < t1.Depth(); ++d)
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
                    CudaDeviceVariable<float>(result.GetDeviceVar(), d * result.GetShape().Dim0Dim1 + b * result.BatchLength()).GetDevicePtr(),
                    n));
            }
        }
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::Transpose(const Tensor& input, Tensor& result) const
    {
        input.CopyToDevice();
        result.CopyToDevice();

        int m = input.Height();
        int n = input.Width();

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
                CudaDeviceVariable<float>(result.GetDeviceVar(), b * result.GetShape().Dim0Dim1).GetDevicePtr(), m));
        }
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::Conv2D(const Tensor& input, const Tensor& kernels, int stride, int paddingX, int paddingY, Tensor& output) const
    {
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
        cudnnGetConvolutionForwardAlgorithm(s_CudnnHandle, inputDesc, kernelsDesc, convolutionDesc, outputDesc, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo);

        size_t workspaceSize;
        cudnnGetConvolutionForwardWorkspaceSize(s_CudnnHandle, inputDesc, kernelsDesc, convolutionDesc, outputDesc, algo, &workspaceSize);
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
    void TensorOpGpu::Conv2DInputGradient(const Tensor& gradient, const Tensor& kernels, int stride, int paddingX, int paddingY, Tensor& inputGradients) const
    {
        gradient.CopyToDevice();
        kernels.CopyToDevice();
        inputGradients.CopyToDevice();

        cudnnConvolutionDescriptor_t convolutionDesc; cudnnCreateConvolutionDescriptor(&convolutionDesc);
        cudnnTensorDescriptor_t gradientDesc; cudnnCreateTensorDescriptor(&gradientDesc);
        cudnnFilterDescriptor_t kernelsDesc; cudnnCreateFilterDescriptor(&kernelsDesc);
        cudnnTensorDescriptor_t inputGradientsDesc; cudnnCreateTensorDescriptor(&inputGradientsDesc);
        
        cudnnSetConvolution2dDescriptor(convolutionDesc, paddingY, paddingX, stride, stride, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
        cudnnSetTensor4dDescriptor(gradientDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, gradient.GetShape().Dimensions[3], gradient.GetShape().Dimensions[2], gradient.GetShape().Dimensions[1], gradient.GetShape().Dimensions[0]);
        cudnnSetFilter4dDescriptor(kernelsDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, kernels.GetShape().Dimensions[3], kernels.GetShape().Dimensions[2], kernels.GetShape().Dimensions[1], kernels.GetShape().Dimensions[0]);
        cudnnSetTensor4dDescriptor(inputGradientsDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, inputGradients.GetShape().Dimensions[3], inputGradients.GetShape().Dimensions[2], inputGradients.GetShape().Dimensions[1], inputGradients.GetShape().Dimensions[0]);

        cudnnConvolutionBwdDataAlgo_t algo;
        cudnnGetConvolutionBackwardDataAlgorithm(s_CudnnHandle, kernelsDesc, gradientDesc, convolutionDesc, inputGradientsDesc, CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, &algo);

        size_t workspaceSize;
        cudnnGetConvolutionBackwardDataWorkspaceSize(s_CudnnHandle, kernelsDesc, gradientDesc, convolutionDesc, inputGradientsDesc, algo, &workspaceSize);
        inputGradients.m_GpuData.UpdateWorkspace(inputGradients.m_GpuData.m_ConvBackWorkspace, workspaceSize);

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
            inputGradients.m_GpuData.m_ConvBackWorkspace->GetDevicePtr(),
            workspaceSize, 
            &beta, 
            inputGradientsDesc, 
            inputGradients.GetDevicePtr()));
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::Conv2DKernelsGradient(const Tensor& input, const Tensor& gradient, int stride, int paddingX, int paddingY, Tensor& kernelsGradient) const
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
    void TensorOpGpu::Pool2D(const Tensor& input, int filterSize, int stride, EPoolingMode type, int paddingX, int paddingY, Tensor& output) const
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
    void TensorOpGpu::Pool2DGradient(const Tensor& output, const Tensor& input, const Tensor& outputGradient, int filterSize, int stride, EPoolingMode type, int paddingX, int paddingY, Tensor& result) const
    {
        output.CopyToDevice();
        input.CopyToDevice();
        outputGradient.CopyToDevice();
        result.CopyToDevice();

        cudnnPoolingDescriptor_t poolingDesc; cudnnCreatePoolingDescriptor(&poolingDesc);
        cudnnTensorDescriptor_t outputDesc; cudnnCreateTensorDescriptor(&outputDesc);
        cudnnTensorDescriptor_t inputDesc; cudnnCreateTensorDescriptor(&inputDesc);
        cudnnTensorDescriptor_t outputGradientDesc; cudnnCreateTensorDescriptor(&outputGradientDesc);
        cudnnTensorDescriptor_t resultDesc; cudnnCreateTensorDescriptor(&resultDesc);

        cudnnSetPooling2dDescriptor(poolingDesc, GetCudnnPoolType(type), CUDNN_NOT_PROPAGATE_NAN, filterSize, filterSize, paddingX, paddingY, stride, stride);
        cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, output.GetShape().Dimensions[3], output.GetShape().Dimensions[2], output.GetShape().Dimensions[1], output.GetShape().Dimensions[0]);
        cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, input.GetShape().Dimensions[3], input.GetShape().Dimensions[2], input.GetShape().Dimensions[1], input.GetShape().Dimensions[0]);
        cudnnSetTensor4dDescriptor(outputGradientDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, outputGradient.GetShape().Dimensions[3], outputGradient.GetShape().Dimensions[2], outputGradient.GetShape().Dimensions[1], outputGradient.GetShape().Dimensions[0]);
        cudnnSetTensor4dDescriptor(resultDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, result.GetShape().Dimensions[3], result.GetShape().Dimensions[2], result.GetShape().Dimensions[1], result.GetShape().Dimensions[0]);

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
            resultDesc, 
            result.GetDevicePtr()));
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::BatchNormalization(const Tensor& input, const Tensor& gamma, const Tensor& beta, const Tensor& runningMean, const Tensor& runningVar, Tensor& output) const
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
            CUDNN_BATCHNORM_PER_ACTIVATION,
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
            _EPSILON));
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::BatchNormalizationTrain(const Tensor& input, const Tensor& gamma, const Tensor& beta, float momentum, Tensor& runningMean, Tensor& runningVar, Tensor& saveMean, Tensor& saveInvVariance, Tensor& output) const
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
            CUDNN_BATCHNORM_PER_ACTIVATION,
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
            _EPSILON,
            saveMean.GetDevicePtr(),
            saveInvVariance.GetDevicePtr()));
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::BatchNormalizationGradient(const Tensor& input, const Tensor& gamma, const Tensor& outputGradient, const Tensor& savedMean, const Tensor& savedInvVariance, Tensor& gammaGradient, Tensor& betaGradient, Tensor& inputGradient) const
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

        float alpha = 1, beta = 0;
        CUDA_CHECK(cudnnBatchNormalizationBackward(
            s_CudnnHandle,
            CUDNN_BATCHNORM_PER_ACTIVATION,
            &alpha,
            &beta,
            &alpha,
            &beta,
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
            _EPSILON,
            savedMean.GetDevicePtr(),
            savedInvVariance.GetDevicePtr()));
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::Elu(const Tensor& input, float alpha, Tensor& result) const
    {
        dim3 blocks, threads;
        GetKernelRunParams(input.Length(), blocks, threads);
        input.CopyToDevice();
        result.CopyToDevice();

        CudaKernels::Elu(blocks, threads, input.Length(), input.GetDevicePtr(), alpha, result.GetDevicePtr());
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::EluGradient(const Tensor& output, const Tensor& outputGradient, float alpha, Tensor& result) const
    {
        dim3 blocks, threads;
        GetKernelRunParams(output.Length(), blocks, threads);
        output.CopyToDevice();
        outputGradient.CopyToDevice();
        result.CopyToDevice();

        CudaKernels::EluGradient(blocks, threads, output.Length(), output.GetDevicePtr(), outputGradient.GetDevicePtr(), alpha, result.GetDevicePtr());
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::Softmax(const Tensor& input, Tensor& result) const
    {
        input.CopyToDevice();
        result.CopyToDevice();

        cudnnTensorDescriptor_t inputDesc; cudnnCreateTensorDescriptor(&inputDesc);
        cudnnTensorDescriptor_t resultDesc; cudnnCreateTensorDescriptor(&resultDesc);

        int n = input.Batch(), c = input.Height(), h = input.Depth(), w = input.Width(); // cuDNN expects values to be in Channel so we need to fake 'reshape' our tensor

        cudnnSetTensor4dDescriptor(inputDesc,CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);
        cudnnSetTensor4dDescriptor(resultDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);

        float alpha = 1, beta = 0;
        CUDA_CHECK(cudnnSoftmaxForward(
            s_CudnnHandle, 
            CUDNN_SOFTMAX_ACCURATE, 
            CUDNN_SOFTMAX_MODE_CHANNEL, 
            &alpha, 
            inputDesc, 
            input.GetDevicePtr(), 
            &beta, 
            resultDesc, 
            result.GetDevicePtr()));
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::SoftmaxGradient(const Tensor& output, const Tensor& outputGradient, Tensor& result) const
    {
        output.CopyToDevice();
        outputGradient.CopyToDevice();
        result.CopyToDevice();

        cudnnTensorDescriptor_t outputDesc; cudnnCreateTensorDescriptor(&outputDesc);
        cudnnTensorDescriptor_t outputGradientDesc; cudnnCreateTensorDescriptor(&outputGradientDesc);
        cudnnTensorDescriptor_t resultDesc; cudnnCreateTensorDescriptor(&resultDesc);

        int n = output.Batch(), c = output.Height(), h = output.Depth(), w = output.Width(); // cuDNN expects values to be in Channel so we need to fake 'reshape' our tensor

        cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);
        cudnnSetTensor4dDescriptor(outputGradientDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);
        cudnnSetTensor4dDescriptor(resultDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);

        float alpha = 1, beta = 0;
        CUDA_CHECK(cudnnSoftmaxBackward(
            s_CudnnHandle, 
            CUDNN_SOFTMAX_ACCURATE, 
            CUDNN_SOFTMAX_MODE_CHANNEL, 
            &alpha, 
            outputDesc, 
            output.GetDevicePtr(), 
            outputGradientDesc, 
            outputGradient.GetDevicePtr(), 
            &beta, 
            resultDesc, 
            result.GetDevicePtr()));
    }

    //////////////////////////////////////////////////////////////////////////
    cudnnPoolingMode_t TensorOpGpu::GetCudnnPoolType(EPoolingMode type)
    {
        if (type == EPoolingMode::Max)
            return CUDNN_POOLING_MAX;
        return CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
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