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
#include "Pix2Pix.h"

#include <cuda.h>
#include <cudnn.h>
#include "Tensors/TensorOpGpu.h"
#include "Tensors/Cuda/CudaErrorCheck.h"

int main()
{
    Tensor::SetForcedOpMode(CPU);

    Tensor img("e:/Dropbox/!BLOG/4.jpg", false);
    Tensor edges = CannyEdgeDetection(img);
    edges.SaveAsImage("edges.png", false);

    //const int DEPTH_IN = 64;
    //const int KERNELS_NUM = 3;

    //Tensor input(Shape(512, 512, DEPTH_IN, 6)); input.FillWithRand();
    //Tensor kernels(Shape(3, 3, DEPTH_IN, KERNELS_NUM)); kernels.FillWithRand();
    //Tensor output(Tensor::GetConvOutputShape(input.GetShape(), KERNELS_NUM, 3, 3, 1, 0, 0, NCHW));
    ////Tensor output(Tensor::GetPooling2DOutputShape(input.GetShape(), 3, 3, 1, 0, 0, NCHW));
    //input.CopyToDevice();
    //kernels.CopyToDevice();
    //output.OverrideDevice();

    //Tensor x(output.GetShape()); x.FillWithRand();
    //Tensor y(output.GetShape()); y.FillWithRand(-1, 1, 5);

    //x.CopyToDevice();
    //y.CopyToDevice();
    //output.OverrideDevice();

    //void* workspacePtr = nullptr;

    //for (int i = 0; i < 40; ++i)
    //{
    //    AutoStopwatch p1(Microseconds);
    //    x.Div(y, output);
    //    cout << "Div " << p1.ToString() << endl;

    //    AutoStopwatch p2(Microseconds);
    //    input.Conv2D(kernels, 1, 0, NCHW, output);
    //    cout << "Other " << p2.ToString() << endl;

    //    AutoStopwatch p3(Microseconds);
    //    output.SyncToHost();
    //    cout << "Sync " << p3.ToString() << endl;
    //}

    ////cin.get();

    //return 0;

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
    //AdaptiveStyleTransfer().Run();
    //AdaptiveStyleTransfer().Test();
    Pix2Pix().Run();

    return 0;
}
