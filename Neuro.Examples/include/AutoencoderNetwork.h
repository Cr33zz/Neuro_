#include <iostream>
#include <string>
#include <vector>

#include "Neuro.h"

using namespace std;
using namespace Neuro;

class AutoencoderNetwork
{
public:
    static void Run()
    {
        Tensor::SetDefaultOpMode(EOpMode::MultiCPU);

        //Based on https://www.youtube.com/watch?v=6Lfra0Tym4M

        auto model = new Sequential();
        model->AddLayer(new Conv2D(Shape(28, 28, 1), 5, 10, 1, 0, new ReLU()));
        model->AddLayer(new MaxPooling2D(model->LastLayer(), 2, 2, 1));
        model->AddLayer(new Conv2D(model->LastLayer(), 2, 20, 1, 0, new ReLU()));
        model->AddLayer(new MaxPooling2D(model->LastLayer(), 2, 2, 1));
        model->AddLayer(new UpSampling2D(model->LastLayer(), 2));
        model->AddLayer(new Conv2DTranspose(model->LastLayer(), 2, 20, 1, 0, new ReLU()));
        model->AddLayer(new UpSampling2D(model->LastLayer(), 2));
        model->AddLayer(new Conv2DTranspose(model->LastLayer(), 5, 10, 1, 0, new ReLU()));
        model->AddLayer(new Conv2DTranspose(model->LastLayer(), 1, 3, 1, 0, new Sigmoid()));

        cout << model->Summary();

        auto net = new NeuralNetwork(model, "autoencoder");
        net->Optimize(new Adam(), new BinaryCrossEntropy());

        Tensor input, output;
        LoadMnistData("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte", input, output, false, 1000);

        net->Fit(input, output, 20, 10, nullptr, nullptr, 1, Track::TrainError);

        cout << model->TrainSummary();

        cin.get();
        return;
    }
};
