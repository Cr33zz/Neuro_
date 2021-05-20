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
    Args args = Args(argc, argv);

    ComputationalGraph().Run();
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