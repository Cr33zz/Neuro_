#include <iostream>
#include <string>
#include <vector>

#include "Neuro.h"

using namespace std;
using namespace Neuro;

class MnistConvNetwork
{
public:
    void Run()
    {
        Tensor::SetDefaultOpMode(GPU);

        auto model = Sequential("mnist_conv", 1337);
        model.AddLayer(new Conv2D(Shape(28, 28, 1), 32, 3, 1, 0, new ReLU()));
        model.AddLayer(new MaxPooling2D(2, 2));
        model.AddLayer(new Conv2D(16, 3, 1, 0, new ReLU()));
        model.AddLayer(new MaxPooling2D(2, 2));
        model.AddLayer(new Dropout(0.2f));
        model.AddLayer(new Flatten());
        model.AddLayer(new Dense(128, new ReLU()));
        model.AddLayer(new Dense(10, new Softmax()));        
        model.Optimize(new Adam(), new BinaryCrossEntropy());

        cout << "Example: " << model.Name() << endl;
        cout << model.Summary();

        Tensor input, output;
        LoadMnistData("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte", input, output, true, false, -1);
        Tensor validationInput, validationOutput;
        LoadMnistData("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte", validationInput, validationOutput, true, false, -1);

        model.Fit(input, output, 200, 2, &validationInput, &validationOutput, 2, All);

        cout << model.TrainSummary();

        cin.get();
        return;
    }
};
