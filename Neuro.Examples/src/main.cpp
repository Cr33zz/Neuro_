#include "AutoencoderNetwork.h"
#include "ConvAutoencoderNetwork.h"
#include "ConvNetwork.h"
#include "FlowNetwork.h"
#include "IrisNetwork.h"
#include "MnistConvNetwork.h"
#include "MnistNetwork.h"
#include "GAN.h"
#include "DeepConvGAN.h"
#include "CifarGAN.h"
#include "ComputationalGraph.h"
#include "NeuralStyleTransfer.h"
#include "FastNeuralStyleTransfer.h"
#include "AdaptiveStyleTransfer.h"

#include <cuda.h>
#include <cudnn.h>
#include "Tensors/TensorOpGpu.h"
#include "Tensors/Cuda/CudaErrorCheck.h"

int main()
{
    //Tensor::SetForcedOpMode(GPU);

    //const int DEPTH_IN = 64;
    //const int KERNELS_NUM = 3;

    //Tensor input(Shape(512, 512, DEPTH_IN, 6)); input.FillWithRand();
    //Tensor kernels(Shape(3, 3, DEPTH_IN, KERNELS_NUM)); kernels.FillWithRand();
    //Tensor output(Tensor::GetConvOutputShape(input.GetShape(), KERNELS_NUM, 3, 3, 1, 0, 0, NCHW));

    //const int ITERS = 10;

    //for (int i = 0; i < ITERS; ++i)
    //{
    //    AutoStopwatch prof(Microseconds);
    //    input.Conv2D(kernels, 1, 0, NCHW, output);
    //    cout << i << ": " << prof.ToString() << endl;
    //}

    //input.CopyToDevice();
    //kernels.CopyToDevice();
    //output.OverrideDevice();
    //
    //cudnnConvolutionDescriptor_t convolutionDesc; cudnnCreateConvolutionDescriptor(&convolutionDesc);
    //cudnnTensorDescriptor_t inputDesc; cudnnCreateTensorDescriptor(&inputDesc);
    //cudnnFilterDescriptor_t kernelsDesc; cudnnCreateFilterDescriptor(&kernelsDesc);
    //cudnnTensorDescriptor_t outputDesc; cudnnCreateTensorDescriptor(&outputDesc);

    //cudnnSetConvolution2dDescriptor(convolutionDesc, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
    //cudnnSetFilter4dDescriptor(kernelsDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, kernels.GetShape().Dimensions[3], kernels.GetShape().Dimensions[2], kernels.GetShape().Dimensions[1], kernels.GetShape().Dimensions[0]);
    //cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, input.GetShape().Dimensions[3], input.GetShape().Dimensions[2], input.GetShape().Dimensions[1], input.GetShape().Dimensions[0]);
    //cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, output.GetShape().Dimensions[3], output.GetShape().Dimensions[2], output.GetShape().Dimensions[1], output.GetShape().Dimensions[0]);

    //cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    ////CUDA_CHECK(cudnnGetConvolutionForwardAlgorithm(s_CudnnHandle, inputDesc, kernelsDesc, convolutionDesc, outputDesc, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo));

    //size_t workspaceSize;
    //CUDA_CHECK(cudnnGetConvolutionForwardWorkspaceSize(TensorOpGpu::s_CudnnHandle, inputDesc, kernelsDesc, convolutionDesc, outputDesc, algo, &workspaceSize));
    //void* workspacePtr;
    //DeviceMemoryManager::Default().Allocate(&workspacePtr, workspaceSize, "conv2d_workspace");

    //for (int i = 0; i < ITERS; ++i)
    //{
    //    AutoStopwatch prof(Microseconds);

    //    output.Zero();

    //    float alpha = 1, beta = 0;
    //    CUDA_CHECK(cudnnConvolutionForward(
    //        TensorOpGpu::s_CudnnHandle,
    //        &alpha,
    //        inputDesc,
    //        input.GetDevicePtr(),
    //        kernelsDesc,
    //        kernels.GetDevicePtr(),
    //        convolutionDesc,
    //        algo,
    //        workspacePtr,
    //        workspaceSize,
    //        &beta,
    //        outputDesc,
    //        output.GetDevicePtr()));

    //    cout << i << ": " << prof.ToString() << endl;
    //}

    //DeviceMemoryManager::Default().Free(workspacePtr);


    //ComputationalGraph().Run();
    //IrisNetwork().Run();
    //ConvNetwork().Run();
    //FlowNetwork().Run();
    //MnistConvNetwork().Run();
    //MnistNetwork().Run();
    //AutoencoderNetwork().Run();
    //ConvAutoencoderNetwork().Run();
    //GAN().Run();
    //DeepConvGAN().Run();
    //CifarGAN().RunDiscriminatorTrainTest();
    //CifarGAN().Run();
    //NeuralStyleTransfer().Run();
    //FastNeuralStyleTransfer().Run();
    AdaptiveStyleTransfer().Run();
    //AdaptiveStyleTransfer().Test();

    cin.get();
    return 0;
}
