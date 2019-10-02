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
#include "VGG16.h"
#include "ComputationalGraph.h"

int main()
{
    /*Tensor t(Shape(2, 3, 4, 5));
    t.FillWithRange(1);*/

    /*Tensor::SetForcedOpMode(GPU);
    auto model = Sequential();
    model.AddLayer(new Conv2D(Shape(2, 4, 4), 4, 3, 1, 1, nullptr, NHWC));
    model.AddLayer(new Conv2D(2, 3, 1, 1, nullptr, NHWC));
    model.AddLayer(new MaxPooling2D(2, 2, 0, NHWC));
    model.LoadWeights("sample.h5");*/

    //cout << model.Predict(input)[0]->ToString() << endl;
    //cout << model.Predict(input)[0]->Transposed({ DepthAxis, WidthAxis, HeightAxis }).ToString() << endl;

    /*for (auto weight : model.Weights())
        cout << weight->ToString() << endl;*/

    auto x = new Variable(2);
    auto y = subtract(add(pow(x, 2), x), new Constant(1));

    auto grads = gradients(y, x);

    auto result = Session::Default->Run(grads);
    cout << (*result[0])(0) << endl;



    ComputationalGraph().Run();
    IrisNetwork().Run();
    ConvNetwork().Run();
    FlowNetwork().Run();
    MnistConvNetwork().Run();
    MnistNetwork().Run();
    AutoencoderNetwork().Run();
    ConvAutoencoderNetwork().Run();
    GAN().Run();
    //DeepConvGAN().Run();
    //CifarGAN().RunDiscriminatorTrainTest();
    //CifarGAN().Run();
    //VGG16().Run();

    cin.get();
    return 0;
}
