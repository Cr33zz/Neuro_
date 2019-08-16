#include "Tensors/TensorOpGpu.h"
#include "Tensors/Cuda/CudaDeviceVariable.h"

namespace Neuro
{
    bool TensorOpGpu::s_Initialized = false;    
    cublasHandle_t TensorOpGpu::s_CublasHandle;
    cudnnHandle_t TensorOpGpu::s_CudnnHandle;

    //////////////////////////////////////////////////////////////////////////
    TensorOpGpu::TensorOpGpu()
    {
        if (!s_Initialized)
        {
            s_Initialized = true;

            //_CudaContext = new CudaContext(0, true);
            cublasCreate_v2(&s_CublasHandle);
            cudnnCreate(&s_CudnnHandle);
            //_CudaStream = new CudaStream();
            //_CudnnContext = new CudaDNNContext();
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
        int batches = t1.Depth() * t1.BatchSize();
        float alpha = 1;
        float beta = 0;

        for (int b = 0; b < batches; ++b)
        {
            cublasGemmEx(s_CublasHandle, 
                         transposeT2 ? CUBLAS_OP_T : CUBLAS_OP_N,
                         transposeT1 ? CUBLAS_OP_T : CUBLAS_OP_N,
                         n, m, k,  // trick to convert row major to column major
                         &alpha,
                         CudaDeviceVariable<float>(*t2.m_GpuData.m_DeviceVar, b * t2.GetShape().Dim0Dim1).GetDevicePtr(), CUDA_R_32F, n,
                         CudaDeviceVariable<float>(*t1.m_GpuData.m_DeviceVar, b * t1.GetShape().Dim0Dim1).GetDevicePtr(), CUDA_R_32F, k,
                         &beta,
                         CudaDeviceVariable<float>(*result.m_GpuData.m_DeviceVar, b * result.GetShape().Dim0Dim1).GetDevicePtr(), CUDA_R_32F, n,
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

        //_CudaBlasHandle.GemmBatched(transposeT2 ? Operation.Transpose : Operation.NonTranspose, 
        //                            transposeT1 ? Operation.Transpose : Operation.NonTranspose, 
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

        float alpha = 1;
        float beta = 0;

        cudnnConvolutionForward(s_CudnnHandle, &alpha, tDesc, t.m_GpuData.m_DeviceVar->GetDevicePtr(), kernelsDesc, kernels.m_GpuData.m_DeviceVar->GetDevicePtr(), convolutionDesc, algo, result.m_GpuData.m_ConvWorkspace->GetDevicePtr(), workspaceSize, &beta, resultDesc, result.m_GpuData.m_DeviceVar->GetDevicePtr());
    }

}


