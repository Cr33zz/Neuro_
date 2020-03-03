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
#include "NeuralStyleTransferHD2.h"

int main(int argc, char *argv[])
{
    Tensor::SetForcedOpMode(GPU);

    Tensor a = Tensor(Shape(3, 5)).FillWithRange();
    Tensor b = Tensor(Shape(3, 4)).FillWithRange(2).Transpose();

    Tensor c = a.MatMul(false, b, false);

    Tensor r = Tensor({ 26, 29, 32, 35, 80, 92, 104, 116, 134, 155, 176, 197, 188, 218, 248, 278, 242, 281, 320, 359 }, Shape(4, 5));



    Args args = Args(argc, argv);

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
    //DeepConvGAN().RunDiscriminatorTrainTest();
    //CifarGAN().Run();
    //CifarGAN().RunDiscriminatorTrainTest();
    //NeuralStyleTransfer().Run();
    //NeuralStyleTransferHD2().Run(args);
    //FastNeuralStyleTransfer().Run();
    //AdaptiveStyleTransfer().Run();
    //AdaptiveStyleTransfer().Test();
    //Pix2Pix().Run();
    //Pix2Pix().RunDiscriminatorTrainTest();

    return 0;
}