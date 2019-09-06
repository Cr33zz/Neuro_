#include <iostream>
#include <string>
#include <vector>

#include "Neuro.h"

using namespace std;
using namespace Neuro;

class MnistNetwork
{
public:
    static void Run()
    {
        Tensor::SetDefaultOpMode(EOpMode::GPU);

        auto model = new Sequential();
        model->AddLayer(new Dense(784, 64, new ReLU()));
        //model->AddLayer(new Dropout(model->LastLayer(), 0.2f));
        model->AddLayer(new Dense(model->LastLayer(), 64, new ReLU()));
        //model->AddLayer(new Dropout(model->LastLayer(), 0.2f));
        model->AddLayer(new Dense(model->LastLayer(), 10, new Softmax()));

        cout << model->Summary();

        auto net = new NeuralNetwork(model, "mnist", 1337);
        net->Optimize(new Adam(), new BinaryCrossEntropy());

        Tensor input, output;
        LoadMnistData("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte", input, output, false, 100);
        input.Reshape(Shape(1, -1, 1, input.Batch()));

        net->Fit(input, output, 10, 10, nullptr, nullptr, 2, Track::TrainError | Track::TrainAccuracy);

        cout << model->TrainSummary();

        cin.get();
        return;
    }
};
