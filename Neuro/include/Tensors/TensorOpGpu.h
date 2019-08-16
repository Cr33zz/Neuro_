#pragma once

#include <cuda.h>
#include <cudnn.h>
#include <cublas.h>

#include "Tensors/TensorOpMultiCpu.h"

namespace Neuro
{
    class TensorOpGpu : public TensorOpMultiCpu
    {
    public:
        TensorOpGpu();

        /*public override void Add(float alpha, Tensor t1, float beta, Tensor t2, Tensor result)
        {
            t1.CopyToDevice();
            t2.CopyToDevice();
            result.CopyToDevice();

            if (t2.BatchSize() == t1.BatchSize())
            {
                cublasSgeam()
                _CudaBlasHandle.Geam(Operation.NonTranspose, Operation.NonTranspose, 
                                     t1.Length, 1, 
                                     alpha, 
                                     t1.m_GpuData.DeviceVar, t1.Length, 
                                     t2.m_GpuData.DeviceVar, t2.Length, 
                                     beta, 
                                     result.m_GpuData.DeviceVar, result.Length);
                return;
            }

            for (int n = 0; n < t1.BatchSize(); ++n)
            {
                _CudaBlasHandle.Geam(Operation.NonTranspose, Operation.NonTranspose, 
                                     t1.BatchLength, 1, 
                                     alpha,
                                     new CudaDeviceVariable<float>(t1.m_GpuData.DeviceVar.DevicePointer + n * t1.BatchLength * sizeof(float)), t1.BatchLength,
                                     t2.m_GpuData.DeviceVar, t2.BatchLength, 
                                     beta,
                                     new CudaDeviceVariable<float>(result.m_GpuData.DeviceVar.DevicePointer + n * result.BatchLength * sizeof(float)), result.BatchLength);
            }
        }*/

        virtual void Mul(bool transposeT1, bool transposeT2, const Tensor& t1, const Tensor& t2, Tensor& result) const override;

        //public override void Transpose(Tensor t, Tensor result)
        //{
        //    t.CopyToDevice();
        //    result.CopyToDevice();

        //    var m = t.Height();
        //    var n = t.Width();

        //    //treat depth as batch
        //    int batches = t.Depth() * t.BatchSize();

        //    for (int b = 0; b < batches; ++b)
        //    {
        //        var tPtr = new CudaDeviceVariable<float>(t.m_GpuData.DeviceVar.DevicePointer + b * t.GetShape().Dim0Dim1 * sizeof(float));

        //        _CudaBlasHandle.Geam(Operation.Transpose, 
        //                             Operation.NonTranspose, m, n,  // trick to convert row major to column major
        //                             1.0f,
        //                             tPtr, n,
        //                             tPtr, m, 
        //                             0.0f,
        //                             new CudaDeviceVariable<float>(result.m_GpuData.DeviceVar.DevicePointer + b * result.GetShape().Dim0Dim1 * sizeof(float)), m);
        //    }
        //}

        virtual void Conv2D(const Tensor& t, const Tensor& kernels, int stride, Tensor::EPaddingType padding, Tensor& result) const override;

        //public override void Conv2DInputGradient(Tensor gradient, Tensor kernels, int stride, Tensor.PaddingType padding, Tensor inputGradients)
        //{
        //    int outputWidth = 0, outputHeight = 0, paddingX = 0, paddingY = 0;
        //    Tensor.GetPaddingParams(padding, gradient.Width(), gradient.Height(), kernels.Width(), kernels.Height(), stride, out outputHeight, out outputWidth, out paddingX, out paddingY);

        //    gradient.CopyToDevice();
        //    kernels.CopyToDevice();
        //    inputGradients.CopyToDevice();

        //    using (var convolutionDesc = new ConvolutionDescriptor())
        //    using (var gradientDesc = new TensorDescriptor())
        //    using (var kernelsDesc = new FilterDescriptor())
        //    using (var inputGradientsDesc = new TensorDescriptor())
        //    {
        //        convolutionDesc.SetConvolution2dDescriptor(paddingY, paddingX, stride, stride, 1, 1, cudnnConvolutionMode.CrossCorrelation, CUDNN_DATA_FLOAT);
        //        gradientDesc.SetTensor4dDescriptor(CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, gradient.GetShape().Dimensions[3], gradient.GetShape().Dimensions[2], gradient.GetShape().Dimensions[1], gradient.GetShape().Dimensions[0]);
        //        kernelsDesc.SetFilter4dDescriptor(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, kernels.GetShape().Dimensions[3], kernels.GetShape().Dimensions[2], kernels.GetShape().Dimensions[1], kernels.GetShape().Dimensions[0]);
        //        inputGradientsDesc.SetTensor4dDescriptor(CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, inputGradients.GetShape().Dimensions[3], inputGradients.GetShape().Dimensions[2], inputGradients.GetShape().Dimensions[1], inputGradients.GetShape().Dimensions[0]);

        //        var algo = cudnnGetConvolutionBackwardDataAlgorithm(kernelsDesc, gradientDesc, convolutionDesc, inputGradientsDesc, cudnnConvolutionBwdDataPreference.PreferFastest, IntPtr.Zero);
        //        var workspaceSize = cudnnGetConvolutionBackwardDataWorkspaceSize(kernelsDesc, gradientDesc, convolutionDesc, inputGradientsDesc, algo);
        //        workspaceSize = workspaceSize == 0 ? new SizeT(1) : workspaceSize;

        //        if (inputGradients.m_GpuData.ConvBackWorkspace == null || inputGradients.m_GpuData.ConvBackWorkspace.Size != workspaceSize)
        //            inputGradients.m_GpuData.ConvBackWorkspace = new CudaDeviceVariable<byte>(workspaceSize);

        //        cudnnConvolutionBackwardData(1.0f, kernelsDesc, kernels.m_GpuData.DeviceVar, gradientDesc, gradient.m_GpuData.DeviceVar, convolutionDesc, algo, inputGradients.m_GpuData.ConvBackWorkspace, 0.0f, inputGradientsDesc, inputGradients.m_GpuData.DeviceVar);
        //    }
        //}

        //public override void Conv2DKernelsGradient(Tensor input, Tensor gradient, int stride, Tensor.PaddingType padding, Tensor kernelsGradient)
        //{
        //    int outputWidth = 0, outputHeight = 0, paddingX = 0, paddingY = 0;
        //    Tensor.GetPaddingParams(padding, input.Width(), input.Height(), kernelsGradient.Width(), kernelsGradient.Height(), stride, out outputHeight, out outputWidth, out paddingX, out paddingY);

        //    gradient.CopyToDevice();
        //    input.CopyToDevice();
        //    kernelsGradient.CopyToDevice();

        //    using (var convolutionDesc = new ConvolutionDescriptor())
        //    using (var gradientDesc = new TensorDescriptor())
        //    using (var inputDesc = new TensorDescriptor())
        //    using (var kernelsGradientsDesc = new FilterDescriptor())
        //    {
        //        convolutionDesc.SetConvolution2dDescriptor(paddingY, paddingX, stride, stride, 1, 1, cudnnConvolutionMode.CrossCorrelation, CUDNN_DATA_FLOAT);
        //        gradientDesc.SetTensor4dDescriptor(CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, gradient.GetShape().Dimensions[3], gradient.GetShape().Dimensions[2], gradient.GetShape().Dimensions[1], gradient.GetShape().Dimensions[0]);
        //        inputDesc.SetTensor4dDescriptor(CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, input.GetShape().Dimensions[3], input.GetShape().Dimensions[2], input.GetShape().Dimensions[1], input.GetShape().Dimensions[0]);
        //        kernelsGradientsDesc.SetFilter4dDescriptor(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, kernelsGradient.GetShape().Dimensions[3], kernelsGradient.GetShape().Dimensions[2], kernelsGradient.GetShape().Dimensions[1], kernelsGradient.GetShape().Dimensions[0]);

        //        var algo = cudnnGetConvolutionBackwardFilterAlgorithm(inputDesc, gradientDesc, convolutionDesc, kernelsGradientsDesc, cudnnConvolutionBwdFilterPreference.PreferFastest, IntPtr.Zero);
        //        var workspaceSize = cudnnGetConvolutionBackwardFilterWorkspaceSize(inputDesc, gradientDesc, convolutionDesc, kernelsGradientsDesc, algo);
        //        workspaceSize = workspaceSize == 0 ? new SizeT(1) : workspaceSize;

        //        if (kernelsGradient.m_GpuData.ConvBackKernelWorkspace == null || kernelsGradient.m_GpuData.ConvBackKernelWorkspace.Size != workspaceSize)
        //            kernelsGradient.m_GpuData.ConvBackKernelWorkspace = new CudaDeviceVariable<byte>(workspaceSize);

        //        cudnnConvolutionBackwardFilter(1.0f, inputDesc, input.m_GpuData.DeviceVar, gradientDesc, gradient.m_GpuData.DeviceVar, convolutionDesc, algo, kernelsGradient.m_GpuData.ConvBackKernelWorkspace, 0.0f, kernelsGradientsDesc, kernelsGradient.m_GpuData.DeviceVar);
        //    }
        //}

        //private cudnnPoolingMode TensorPoolTypeToCuDNNPoolType(Tensor.PoolType type)
        //{
        //    if (type == Tensor.PoolType.Max)
        //        return cudnnPoolingMode.Max;
        //    return cudnnPoolingMode.AverageCountIncludePadding;
        //}

        //public override void Pool(Tensor t, int filterSize, int stride, Tensor.PoolType type, int paddingX, int paddingY, Tensor result)
        //{
        //    t.CopyToDevice();
        //    result.CopyToDevice();

        //    using (var poolingDesc = new PoolingDescriptor())
        //    using (var tDesc = new TensorDescriptor())
        //    using (var resultDesc = new TensorDescriptor())
        //    {
        //        poolingDesc.SetPooling2dDescriptor(TensorPoolTypeToCuDNNPoolType(type), cudnnNanPropagation.NotPropagateNan, filterSize, filterSize, paddingX, paddingY, stride, stride);
        //        tDesc.SetTensor4dDescriptor(CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, t.GetShape().Dimensions[3], t.GetShape().Dimensions[2], t.GetShape().Dimensions[1], t.GetShape().Dimensions[0]);
        //        resultDesc.SetTensor4dDescriptor(CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, result.GetShape().Dimensions[3], result.GetShape().Dimensions[2], result.GetShape().Dimensions[1], result.GetShape().Dimensions[0]);

        //        cudnnPoolingForward(poolingDesc, 1.0f, tDesc, t.m_GpuData.DeviceVar, 0.0f, resultDesc, result.m_GpuData.DeviceVar);
        //    }
        //}

        //public override void PoolGradient(Tensor output, Tensor input, Tensor outputGradient, int filterSize, int stride, Tensor.PoolType type, int paddingX, int paddingY, Tensor result)
        //{
        //    output.CopyToDevice();
        //    input.CopyToDevice();
        //    outputGradient.CopyToDevice();
        //    result.CopyToDevice();

        //    using (var poolingDesc = new PoolingDescriptor())
        //    using (var outputDesc = new TensorDescriptor())
        //    using (var inputDesc = new TensorDescriptor())
        //    using (var outputGradientDesc = new TensorDescriptor())
        //    using (var resultDesc = new TensorDescriptor())
        //    {
        //        poolingDesc.SetPooling2dDescriptor(TensorPoolTypeToCuDNNPoolType(type), cudnnNanPropagation.NotPropagateNan, filterSize, filterSize, paddingX, paddingY, stride, stride);
        //        outputDesc.SetTensor4dDescriptor(CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, output.GetShape().Dimensions[3], output.GetShape().Dimensions[2], output.GetShape().Dimensions[1], output.GetShape().Dimensions[0]);
        //        inputDesc.SetTensor4dDescriptor(CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, input.GetShape().Dimensions[3], input.GetShape().Dimensions[2], input.GetShape().Dimensions[1], input.GetShape().Dimensions[0]);
        //        outputGradientDesc.SetTensor4dDescriptor(CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, outputGradient.GetShape().Dimensions[3], outputGradient.GetShape().Dimensions[2], outputGradient.GetShape().Dimensions[1], outputGradient.GetShape().Dimensions[0]);
        //        resultDesc.SetTensor4dDescriptor(CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, result.GetShape().Dimensions[3], result.GetShape().Dimensions[2], result.GetShape().Dimensions[1], result.GetShape().Dimensions[0]);

        //        cudnnPoolingBackward(poolingDesc, 1.0f, outputDesc, output.m_GpuData.DeviceVar, outputGradientDesc, outputGradient.m_GpuData.DeviceVar, inputDesc, input.m_GpuData.DeviceVar, 0.0f, resultDesc, result.m_GpuData.DeviceVar);
        //    }
        //}

        //public override void SumBatches(Tensor t, Tensor result)
        //{
        //    t.CopyToDevice();
        //    result.CopyToDevice();

        //    int batchLen = t.BatchLength;

        //    for (int n = 0; n < t.BatchSize(); ++n)
        //    {
        //        _CudaBlasHandle.Geam(Operation.NonTranspose, Operation.NonTranspose,
        //                             batchLen, 1,
        //                             1.0f,
        //                             new CudaDeviceVariable<float>(t.m_GpuData.DeviceVar.DevicePointer + n * batchLen * sizeof(float)), batchLen,
        //                             result.m_GpuData.DeviceVar, batchLen,
        //                             1.0f,
        //                             result.m_GpuData.DeviceVar, batchLen);
        //    }
        //}

        //public override void Elu(Tensor input, float alpha, Tensor result)
        //{
        //    _KernelLoader.RunKernel("elu", input, result, new object[] { alpha });
        //}

        //public override void EluGradient(Tensor output, Tensor outputGradient, float alpha, Tensor result)
        //{
        //    _KernelLoader.RunKernel("elu_grad", output, outputGradient, result, new object[] { alpha });
        //}

        //public override void Softmax(Tensor input, Tensor result)
        //{
        //    input.CopyToDevice();
        //    result.CopyToDevice();

        //    using (var inputDesc = new TensorDescriptor())
        //    using (var resultDesc = new TensorDescriptor())
        //    {
        //        int n = input.BatchSize(), c = input.Height(), h = input.Depth(), w = input.Width(); // cuDNN expects values to be in Channel so we need to fake 'reshape' our tensor

        //        inputDesc.SetTensor4dDescriptor(CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);
        //        resultDesc.SetTensor4dDescriptor(CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);

        //        cudnnSoftmaxForward(cudnnSoftmaxAlgorithm.Accurate, cudnnSoftmaxMode.Channel, 1.0f, inputDesc, input.m_GpuData.DeviceVar, 0.0f, resultDesc, result.m_GpuData.DeviceVar);
        //    }
        //}

        //public override void SoftmaxGradient(Tensor output, Tensor outputGradient, Tensor result)
        //{
        //    output.CopyToDevice();
        //    outputGradient.CopyToDevice();
        //    result.CopyToDevice();

        //    using (var outputDesc = new TensorDescriptor())
        //    using (var outputGradientDesc = new TensorDescriptor())
        //    using (var resultDesc = new TensorDescriptor())
        //    {
        //        int n = output.BatchSize(), c = output.Height(), h = output.Depth(), w = output.Width(); // cuDNN expects values to be in Channel so we need to fake 'reshape' our tensor

        //        outputDesc.SetTensor4dDescriptor(CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);
        //        outputGradientDesc.SetTensor4dDescriptor(CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);
        //        resultDesc.SetTensor4dDescriptor(CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);

        //        cudnnSoftmaxBackward(cudnnSoftmaxAlgorithm.Accurate, cudnnSoftmaxMode.Channel, 1.0f, outputDesc, output.m_GpuData.DeviceVar, outputGradientDesc, outputGradient.m_GpuData.DeviceVar, 0.0f, resultDesc, result.m_GpuData.DeviceVar);
        //    }
        //}

    private:
        static bool s_Initialized;
        //static CudaContext s_CudaContext;
        //static cudaStream_t s_CudaStream = nullptr;
        static cublasHandle_t s_CublasHandle;
        static cudnnHandle_t s_CudnnHandle;
        //static KernelLoader _KernelLoader;
    };
}
