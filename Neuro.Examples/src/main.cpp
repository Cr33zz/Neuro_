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
    Tensor::SetForcedOpMode(GPU);
    cin.get();

    {
        auto trainFiles = LoadFilesList("e:/Downloads/flowers", false, true);

        const Shape IMG_SHAPE(256, 256, 3);
        const uint32_t PATCH_SIZE = 64;
        const uint32_t BATCH_SIZE = 4;

        Tensor condImages(Shape::From(IMG_SHAPE, BATCH_SIZE), "cond_image");
        Tensor realImages(Shape::From(IMG_SHAPE, BATCH_SIZE), "output_image");

        Pix2Pix::Pix2PixImageLoader loader(trainFiles, BATCH_SIZE, 1, 1337);
        DataPreloader preloader({ &condImages, &realImages }, { &loader }, 5, false);

        //cin.get();
        for (int i = 0; i < 2000; ++i)
        {
            /*Tensor t(Shape(256,256,3,1), "t");
            t.FillWithRand(-1, 0, 255);
            Tensor e = CannyEdgeDetection(t);
            e.Name("e");
            cout << e.DataPtrUnsafe()[0] << endl;*/
            preloader.Load();

            /*if (i == 2)
            {
                DumpMemoryManagers("mem_1.log");
                cin.get();
            }*/
        }
    }
    ReleaseAllMemory();
    DumpMemoryManagers("mem_2.log");
    cin.get();

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
