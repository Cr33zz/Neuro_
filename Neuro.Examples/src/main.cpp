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
    /*cin.get();
    for (int i = 0; i < 100; ++i)
    {
        Tensor t(Shape(256,256,3,1));
        t.FillWithRand(-1, 0, 255);
        Tensor e = CannyEdgeDetection(t);
        cout << e.DataPtrUnsafe()[0] << endl;
    }
    cin.get();*/

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
    //FastNeuralStyleTransfer().Run();
    //AdaptiveStyleTransfer().Run();
    //AdaptiveStyleTransfer().Test();
    Pix2Pix().Run();
    //Pix2Pix().RunDiscriminatorTrainTest();

    return 0;
}
