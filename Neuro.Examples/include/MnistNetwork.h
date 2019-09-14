#include <iostream>
#include <string>
#include <vector>

#include "Neuro.h"

using namespace std;
using namespace Neuro;

class MnistNetwork
{
public:
    void Run()
    {
        Tensor::SetDefaultOpMode(GPU);

        auto model = Sequential("mnist", 1337);
        model.AddLayer(new Dense(784, 64, new ReLU()));
        //model.AddLayer(new BatchNormalization());
        model.AddLayer(new Dropout(0.2f));
        model.AddLayer(new Dense(64, new ReLU()));
        //model.AddLayer(new BatchNormalization());
        model.AddLayer(new Dropout(0.2f));
        model.AddLayer(new Dense(10, new Softmax()));

        cout << "Example: " << model.Name() << endl;
        cout << model.Summary();

        model.Optimize(new Adam(), new BinaryCrossEntropy());

        Tensor input, output;
        LoadMnistData("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte", input, output, true, false, -1);
        input.Reshape(Shape(1, -1, 1, input.Batch()));
        Tensor validationInput, validationOutput;
        LoadMnistData("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte", validationInput, validationOutput, true, false, -1);
        validationInput.Reshape(Shape(1, -1, 1, validationInput.Batch()));

        model.Fit(input, output, 128, 10, &validationInput, &validationOutput, 2, All);

        cout << model.TrainSummary();

        cin.get();
        return;
    }
};
