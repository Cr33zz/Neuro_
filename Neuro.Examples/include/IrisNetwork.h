#include <iostream>
#include <string>
#include <vector>

#include "Neuro.h"

using namespace std;
using namespace Neuro;

class IrisNetwork
{
public:
    void Run()
    {
        Tensor::SetDefaultOpMode(GPU);
        GlobalRngSeed(1337);

        auto model = Sequential("iris");
        model.AddLayer(new Dense(4, 1000, new ReLU()));
        model.AddLayer(new Dense(model.LastLayer(), 500, new ReLU()));
        model.AddLayer(new Dense(model.LastLayer(), 300, new ReLU()));

        //auto dropout = new Dropout(model.LastLayer(), 0.2f);
        //dropout->SetTrainable(false);

        //model.AddLayer(dropout);
        model.AddLayer(new Dense(model.LastLayer(), 3, new Softmax()));
        model.Optimize(new Adam(), new BinaryCrossEntropy());

        cout << "Example: " << model.Name() << endl;
        cout << model.Summary();

        Tensor inputs;
        Tensor outputs;
        LoadCSVData("data/iris_data.csv", 3, inputs, outputs, true);
        inputs = inputs.Normalized(BatchAxis);

        model.Fit(inputs, outputs, 40, 20, nullptr, nullptr, 2, TrainError|TrainAccuracy);

        cout << model.TrainSummary();

        cin.get();
        return;
    }
};
