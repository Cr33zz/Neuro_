#include <iostream>
#include <string>
#include <vector>

#include "Neuro.h"

using namespace std;
using namespace Neuro;

class MnistConvNetwork
{
public:
    static void Run()
    {
        Tensor::SetDefaultOpMode(EOpMode::MultiCPU);

        auto model = new Sequential();
        model->AddLayer(new Conv2D(Shape(28, 28, 1), 3, 28, 1, 0, new ReLU()));
        model->AddLayer(new MaxPooling2D(model->LastLayer(), 2));
        model->AddLayer(new Flatten(model->LastLayer()));
        model->AddLayer(new Dense(model->LastLayer(), 128, new ReLU()));
        model->AddLayer(new Dropout(model->LastLayer(), 0.2f));
        model->AddLayer(new Dense(model->LastLayer(), 10, new Softmax()));

        cout << model->Summary();

        auto net = new NeuralNetwork(model, "mnist_conv");
        net->Optimize(new Adam(), new BinaryCrossEntropy());

        Tensor input, output;
        LoadMnistData("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte", input, output, false, 10000);

        net->Fit(input, output, 50, 10, nullptr, nullptr, 2, Track::TrainError | Track::TrainAccuracy);

        cout << model->TrainSummary();

        cin.get();
        return;
    }
};
