#include <iostream>
#include <string>
#include <vector>

#include "Neuro.h"
#include "Debug.h"

using namespace std;
using namespace Neuro;

class IrisNetwork
{
public:
    void Run()
    {
        //Debug::LogAllGrads();
        //Debug::LogAllOutputs();
        //Debug::LogOutput("loss/cross_entropy/negative");
        
        Tensor::SetDefaultOpMode(GPU);
        GlobalRngSeed(1337);

        auto model = Sequential("iris");
        model.AddLayer(new Dense(4, 1000, new ReLU()));
        model.AddLayer(new Dense(500, new ReLU()));
        model.AddLayer(new Dense(300, new ReLU()));

        /*auto dropout = new Dropout(model.LastLayer(), 0.2f);
        dropout->SetTrainable(false);

        model.AddLayer(dropout);*/
        model.AddLayer(new Dense(3, new Softmax()));
        model.Optimize(new Adam(), new BinaryCrossEntropy(), Loss|Accuracy);

        cout << "Example: " << model.Name() << endl;
        cout << model.Summary();

        Tensor inputs;
        Tensor outputs;
        LoadCSVData("data/iris_data.csv", 3, inputs, outputs, true);
        inputs = inputs.Normalized(BatchAxis);

        model.Fit(inputs, outputs, 40, 20, nullptr, nullptr, 2);

        cout << model.TrainSummary();

        cin.get();
        return;
    }
};
