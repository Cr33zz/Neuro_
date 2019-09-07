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
        model->AddLayer(new Dropout(model->LastLayer(), 0.2f));
        model->AddLayer(new Dense(model->LastLayer(), 64, new ReLU()));
        model->AddLayer(new Dropout(model->LastLayer(), 0.2f));
        model->AddLayer(new Dense(model->LastLayer(), 10, new Softmax()));

        cout << model->Summary();

        auto net = new NeuralNetwork(model, "mnist", 1337);
        net->Optimize(new Adam(), new BinaryCrossEntropy());

        Tensor input, output;
        LoadMnistData("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte", input, output, false, 6000);
        input.Reshape(Shape(1, -1, 1, input.Batch()));
        Tensor validationInput, validationOutput;
        LoadMnistData("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte", validationInput, validationOutput, false, 1000);
        validationInput.Reshape(Shape(1, -1, 1, validationInput.Batch()));

        net->Fit(input, output, 128, 10, &validationInput, &validationOutput, 2, Track::All);

        cout << model->TrainSummary();

        cin.get();
        return;
    }
};
