#include "Tensors/TensorOpGpu.h"
#include "Tensors/Cuda/CudaDeviceVariable.h"
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

                //cout << "GPU: " << s_CudaDevProp.name << "(tpb: " << s_CudaDevProp.maxThreadsPerBlock << ")\n";
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
            cublasSgeam(s_CublasHandle,
                        CUBLAS_OP_N, CUBLAS_OP_N,
                        t1.Length(), 1,
                        &alpha,
                        t1.GetDevicePtr(), t1.Length(),
                        &beta,
                        t2.GetDevicePtr(), t2.Length(),
                        result.GetDevicePtr(), result.Length());
            return;
        }

        for (int n = 0; n < t1.Batch(); ++n)
        {
            cublasSgeam(s_CublasHandle,
                        CUBLAS_OP_N, CUBLAS_OP_N,
                        t1.BatchLength(), 1,
                        &alpha,
                        CudaDeviceVariable<float>(t1.GetDeviceVar(), n * t1.BatchLength()).GetDevicePtr(), t1.BatchLength(),
                        &beta,
                        t2.GetDevicePtr(), t2.BatchLength(),
                        CudaDeviceVariable<float>(result.GetDeviceVar(), n * result.BatchLength()).GetDevicePtr(), result.BatchLength());
        }
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::Mul(bool transposeT1, bool transposeT2, const Tensor& t1, const Tensor& t2, Tensor& result) const
    {
        t1.CopyToDevice();
        t2.CopyToDevice();
        result.CopyToDevice();

        int m = t1.Height();
        int n = t2.Width();
        int k = t1.Width();

        //treat depth as batch
        int batches = t1.Depth() * t1.Batch();
        float alpha = 1, beta = 0;

        for (int b = 0; b < batches; ++b)
        {
            cublasGemmEx(s_CublasHandle, 
                         transposeT2 ? CUBLAS_OP_T : CUBLAS_OP_N,
                         transposeT1 ? CUBLAS_OP_T : CUBLAS_OP_N,
                         n, m, k,  // trick to convert row major to column major
                         &alpha,
                         CudaDeviceVariable<float>(t2.GetDeviceVar(), b * t2.GetShape().Dim0Dim1).GetDevicePtr(), CUDA_R_32F, n,
                         CudaDeviceVariable<float>(t1.GetDeviceVar(), b * t1.GetShape().Dim0Dim1).GetDevicePtr(), CUDA_R_32F, k,
                         &beta,
                         CudaDeviceVariable<float>(result.GetDeviceVar(), b * result.GetShape().Dim0Dim1).GetDevicePtr(), CUDA_R_32F, n,
                         CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
        }

        //CUdeviceptr[] aArray = new CUdeviceptr[batches];
        //CUdeviceptr[] bArray = new CUdeviceptr[batches];
        //CUdeviceptr[] cArray = new CUdeviceptr[batches];

        //for (int b = 0; b < batches; ++b)
        //{
        //    aArray[b] = t1.m_GpuData.DeviceVar.DevicePointer + b * t1.GetShape().Dim0Dim1 * sizeof(float);
        //    bArray[b] = t2.m_GpuData.DeviceVar.DevicePointer + b * t2.GetShape().Dim0Dim1 * sizeof(float);
        //    cArray[b] = result.m_GpuData.DeviceVar.DevicePointer + b * result.GetShape().Dim0Dim1 * sizeof(float);
        //}

        //var dev_aArray = new CudaDeviceVariable<CUdeviceptr>(batches * 4);
        //dev_aArray.CopyToDevice(aArray);
        //var dev_bArray = new CudaDeviceVariable<CUdeviceptr>(batches * 4);
        //dev_bArray.CopyToDevice(bArray);
        //var dev_cArray = new CudaDeviceVariable<CUdeviceptr>(batches * 4);
        //dev_cArray.CopyToDevice(cArray);

        //_CudaBlasHandle.GemmBatched(transposeT2 ? CUBLAS_OP_T : CUBLAS_OP_N, 
        //                            transposeT1 ? CUBLAS_OP_T : CUBLAS_OP_N, 
        //                            n, m, k, 
        //                            1.0f, 
        //                            dev_bArray, n, 
        //                            dev_aArray, k, 
        //                            0.0f, 
        //                            dev_cArray, n, 
        //                            batches);

        //dev_aArray.Dispose();
        //dev_bArray.Dispose();
        //dev_cArray.Dispose();
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::Transpose(const Tensor& t, Tensor& result) const
    {
        t.CopyToDevice();
        result.CopyToDevice();

        int m = t.Height();
        int n = t.Width();

        //treat depth as batch
        int batches = t.Depth() * t.Batch();
        float alpha = 1, beta = 0;

        for (int b = 0; b < batches; ++b)
        {
            CudaDeviceVariable<float> tVar(t.GetDeviceVar(), b * t.GetShape().Dim0Dim1);

            cublasSgeam(s_CublasHandle,
                        CUBLAS_OP_T,
                        CUBLAS_OP_N, m, n,  // trick to convert row major to column major
                        &alpha,
                        tVar.GetDevicePtr(), n,
                        &beta,
                        tVar.GetDevicePtr(), m,
                        CudaDeviceVariable<float>(result.GetDeviceVar(), b * result.GetShape().Dim0Dim1).GetDevicePtr(), m);
        }
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::Conv2D(const Tensor& t, const Tensor& kernels, int stride, Tensor::EPaddingType padding, Tensor& result) const
    {
        int outputWidth = 0, outputHeight = 0, paddingX = 0, paddingY = 0;
        Tensor::GetPaddingParams(padding, t.Width(), t.Height(), kernels.Width(), kernels.Height(), stride, outputHeight, outputWidth, paddingX, paddingY);

        t.CopyToDevice();
        kernels.CopyToDevice();
        result.CopyToDevice();

        cudnnConvolutionDescriptor_t convolutionDesc; cudnnCreateConvolutionDescriptor(&convolutionDesc);
        cudnnTensorDescriptor_t tDesc; cudnnCreateTensorDescriptor(&tDesc);
        cudnnFilterDescriptor_t kernelsDesc; cudnnCreateFilterDescriptor(&kernelsDesc);
        cudnnTensorDescriptor_t resultDesc; cudnnCreateTensorDescriptor(&resultDesc);

        cudnnSetConvolution2dDescriptor(convolutionDesc, paddingY, paddingX, stride, stride, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
        cudnnSetTensor4dDescriptor(tDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, t.GetShape().Dimensions[3], t.GetShape().Dimensions[2], t.GetShape().Dimensions[1], t.GetShape().Dimensions[0]);
        cudnnSetFilter4dDescriptor(kernelsDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, kernels.GetShape().Dimensions[3], kernels.GetShape().Dimensions[2], kernels.GetShape().Dimensions[1], kernels.GetShape().Dimensions[0]);
        cudnnSetTensor4dDescriptor(resultDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, result.GetShape().Dimensions[3], result.GetShape().Dimensions[2], result.GetShape().Dimensions[1], result.GetShape().Dimensions[0]);

        cudnnConvolutionFwdAlgo_t algo;
        cudnnGetConvolutionForwardAlgorithm(s_CudnnHandle, tDesc, kernelsDesc, convolutionDesc, resultDesc, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo);
        size_t workspaceSize;
        cudnnGetConvolutionForwardWorkspaceSize(s_CudnnHandle, tDesc, kernelsDesc, convolutionDesc, resultDesc, algo, &workspaceSize);
        result.m_GpuData.UpdateWorkspace(result.m_GpuData.m_ConvWorkspace, workspaceSize);

        float alpha = 1, beta = 0;
        cudnnConvolutionForward(s_CudnnHandle, &alpha, tDesc, t.GetDevicePtr(), kernelsDesc, kernels.GetDevicePtr(), convolutionDesc, algo, result.m_GpuData.m_ConvWorkspace, workspaceSize, &beta, resultDesc, result.GetDevicePtr());
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::Conv2DInputGradient(const Tensor& gradient, const Tensor& kernels, int stride, Tensor::EPaddingType padding, Tensor& inputGradients) const
    {
        int outputWidth = 0, outputHeight = 0, paddingX = 0, paddingY = 0;
        Tensor::GetPaddingParams(padding, gradient.Width(), gradient.Height(), kernels.Width(), kernels.Height(), stride, outputHeight, outputWidth, paddingX, paddingY);

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
        cudnnConvolutionBackwardData(s_CudnnHandle, &alpha, kernelsDesc, kernels.GetDevicePtr(), gradientDesc, gradient.GetDevicePtr(), convolutionDesc, algo, inputGradients.m_GpuData.m_ConvBackWorkspace, workspaceSize, &beta, inputGradientsDesc, inputGradients.GetDevicePtr());
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::Conv2DKernelsGradient(const Tensor& input, const Tensor& gradient, int stride, Tensor::EPaddingType padding, Tensor& kernelsGradient) const
    {
        int outputWidth = 0, outputHeight = 0, paddingX = 0, paddingY = 0;
        Tensor::GetPaddingParams(padding, input.Width(), input.Height(), kernelsGradient.Width(), kernelsGradient.Height(), stride, outputHeight, outputWidth, paddingX, paddingY);

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
        cudnnConvolutionBackwardFilter(s_CudnnHandle, &alpha, inputDesc, input.GetDevicePtr(), gradientDesc, gradient.GetDevicePtr(), convolutionDesc, algo, kernelsGradient.m_GpuData.m_ConvBackKernelWorkspace, workspaceSize, &beta, kernelsGradientsDesc, kernelsGradient.GetDevicePtr());
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::Pool(const Tensor& t, int filterSize, int stride, Tensor::EPoolType type, int paddingX, int paddingY, Tensor& result) const
    {
        t.CopyToDevice();
        result.CopyToDevice();

        cudnnPoolingDescriptor_t poolingDesc; cudnnCreatePoolingDescriptor(&poolingDesc);
        cudnnTensorDescriptor_t tDesc; cudnnCreateTensorDescriptor(&tDesc);
        cudnnTensorDescriptor_t resultDesc; cudnnCreateTensorDescriptor(&resultDesc);

        cudnnSetPooling2dDescriptor(poolingDesc, GetCudnnPoolType(type), CUDNN_NOT_PROPAGATE_NAN, filterSize, filterSize, paddingX, paddingY, stride, stride);
        cudnnSetTensor4dDescriptor(tDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, t.GetShape().Dimensions[3], t.GetShape().Dimensions[2], t.GetShape().Dimensions[1], t.GetShape().Dimensions[0]);
        cudnnSetTensor4dDescriptor(resultDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, result.GetShape().Dimensions[3], result.GetShape().Dimensions[2], result.GetShape().Dimensions[1], result.GetShape().Dimensions[0]);

        float alpha = 1, beta = 0;
        cudnnPoolingForward(s_CudnnHandle, poolingDesc, &alpha, tDesc, t.GetDevicePtr(), &beta, resultDesc, result.GetDevicePtr());
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::PoolGradient(const Tensor& output, const Tensor& input, const Tensor& outputGradient, int filterSize, int stride, Tensor::EPoolType type, int paddingX, int paddingY, Tensor& result) const
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
        cudnnPoolingBackward(s_CudnnHandle, poolingDesc, &alpha, outputDesc, output.GetDevicePtr(), outputGradientDesc, outputGradient.GetDevicePtr(), inputDesc, input.GetDevicePtr(), &beta, resultDesc, result.GetDevicePtr());
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpGpu::SumBatches(const Tensor& t, Tensor& result) const
    {
        t.CopyToDevice();
        result.CopyToDevice();

        int batchLen = t.BatchLength();
        float alpha = 1, beta = 1;

        for (int n = 0; n < t.Batch(); ++n)
        {
            cublasSgeam(s_CublasHandle,
                        CUBLAS_OP_N, CUBLAS_OP_N,
                        batchLen, 1,
                        &alpha,
                        CudaDeviceVariable<float>(t.GetDeviceVar(), n * batchLen).GetDevicePtr(), batchLen,
                        &beta,
                        result.GetDevicePtr(), batchLen,
                        result.GetDevicePtr(), batchLen);
        }
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
        cudnnSoftmaxForward(s_CudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, &alpha, inputDesc, input.GetDevicePtr(), &beta, resultDesc, result.GetDevicePtr());
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
        cudnnSoftmaxBackward(s_CudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, &alpha, outputDesc, output.GetDevicePtr(), outputGradientDesc, outputGradient.GetDevicePtr(), &beta, resultDesc, result.GetDevicePtr());
    }

    //////////////////////////////////////////////////////////////////////////
    cudnnPoolingMode_t TensorOpGpu::GetCudnnPoolType(Tensor::EPoolType type)
    {
        if (type == Tensor::EPoolType::Max)
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
}