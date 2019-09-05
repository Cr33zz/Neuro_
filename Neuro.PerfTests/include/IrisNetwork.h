#include <iostream>
#include <string>
#include <vector>

#include "Neuro.h"

using namespace std;
using namespace Neuro;

class IrisNetwork
{
public:
    static void Run()
    {
        Tensor::SetDefaultOpMode(EOpMode::MultiCPU);

        auto model = new Sequential();
        model->AddLayer(new Dense(4, 1000, new ReLU()));
        model->AddLayer(new Dense(model->LastLayer(), 500, new ReLU()));
        model->AddLayer(new Dense(model->LastLayer(), 300, new ReLU()));
        model->AddLayer(new Dropout(model->LastLayer(), 0.2f));
        model->AddLayer(new Dense(model->LastLayer(), 3, new Softmax()));

        cout << model->Summary();

        auto net = new NeuralNetwork(model, "iris", 100);
        net->Optimize(new Adam(), new BinaryCrossEntropy());

        Tensor inputs;
        Tensor outputs;
        LoadCSVData("data/iris_data.csv", 3, inputs, outputs, true);
        inputs = inputs.Normalized(EAxis::Feature);

        net->Fit(inputs, outputs, 40, 20, nullptr, nullptr, 2, Track::TrainError | Track::TrainAccuracy);

        cout << model->TrainSummary();

        cin.get();
        return;
    }
};
