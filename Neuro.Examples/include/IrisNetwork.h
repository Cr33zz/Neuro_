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
        Tensor::SetDefaultOpMode(EOpMode::GPU);

        auto model = new Sequential("iris", 100);
        model->AddLayer(new Dense(4, 1000, new ReLU()));
        model->AddLayer(new Dense(model->LastLayer(), 500, new ReLU()));
        model->AddLayer(new Dense(model->LastLayer(), 300, new ReLU()));
        model->AddLayer(new Dropout(model->LastLayer(), 0.2f));
        model->AddLayer(new Dense(model->LastLayer(), 3, new Softmax()));

        cout << model->Summary();

        model->Optimize(new Adam(), new BinaryCrossEntropy());

        Tensor inputs;
        Tensor outputs;
        LoadCSVData("data/iris_data.csv", 3, inputs, outputs, true);
        inputs = inputs.Normalized(EAxis::Feature);

        model->Fit(inputs, outputs, 40, 20, nullptr, nullptr, 2, ETrack::TrainError | ETrack::TrainAccuracy);

        cout << model->TrainSummary();

        cin.get();
        return;
    }
};
