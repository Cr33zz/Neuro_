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
#include "NeuralStyleTransfer.h"
#include "FastNeuralStyleTransfer.h"
#include "AdaptiveStyleTransfer.h"

int main()
{
    /*auto m1In1 = (new Input(Shape(32)))->Outputs();
    auto m1X = (new Dense(10, new Sigmoid()))->Call(m1In1);
    auto m1 = new Flow(m1In1, { m1In1[0], m1X[0] });

    auto m2In1 = (new Input(Shape(32)))->Outputs();
    auto m2X = (new Dense(10, new Sigmoid()))->Call(m2In1);
    auto m2 = new Flow({ m2In1[0] }, { m2X[0] });

    auto seq = new Sequential();
    seq->AddLayer(m2);
    seq->AddLayer(new Dense(10));*/

    

    /*auto m3In1 = new Input(Shape(10));
    auto m3In2 = new Input(Shape(10));
    auto m3X = (new Merge(MergeSum, new Sigmoid()))->Call({ m3In1, m3In2 });
    m3X = (new Dense(5, new Tanh()))->Link(m3X);
    auto m3 = (new Flow({ m3In1, m3In2 }, { m3X }))->Link({ m1->ModelOutputLayers()[1], m2->ModelOutputLayers()[0] });

    auto model = new Flow({ m1->ModelInputLayers()[0], m2->ModelInputLayers()[0] }, m3->ModelOutputLayers());
    model->Optimize(new SGD(0.05f), new MeanSquareError());

    const_tensor_ptr_vec_t inputs = { &(new Tensor(Shape(32)))->FillWithRand(), &(new Tensor(Shape(32)))->FillWithRand() };
    const_tensor_ptr_vec_t outputs = { &(new Tensor(Shape(5)))->FillWithRand() };

    model->Fit(inputs, outputs, 1, 200, nullptr, nullptr, 1, ETrack::TrainError, false);*/

    /*auto m1 = new Sequential("m1");
    m1->AddLayer(new Input(Shape(2)));
    m1->AddLayer(new Dense(3));    
    
    auto m2 = new Sequential("m2");
    m2->AddLayer(new Input(Shape(3)));
    m2->AddLayer(new Dense(2));
    m2->Optimize(new SGD(), new MeanSquareError());

    auto m3 = new Sequential("m3");
    m3->AddLayer(m1);
    m3->AddLayer(m2);
    m3->Optimize(new SGD(), new MeanSquareError());*/

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

        /*auto x = new Variable(2);
        auto y = subtract(add(pow(x, 2), x), new Constant(1));

        auto grads = gradients(y, x);

        auto result = Session::Default()->Run(grads);
        cout << (*result[0])(0) << endl;*/



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
    //VGG16().Run();
    //NeuralStyleTransfer().Run();
    FastNeuralStyleTransfer().Run();
    //FastNeuralStyleTransfer().Test();
    //AdaptiveStyleTransfer().Run();

    cin.get();
    return 0;
}
