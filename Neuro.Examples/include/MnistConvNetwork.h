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
        Tensor::SetDefaultOpMode(EOpMode::GPU);

        auto model = Sequential("mnist_conv", 1337);
        model.AddLayer(new Conv2D(Shape(28, 28, 1), 3, 32, 1, 0, new ReLU()));
        model.AddLayer(new MaxPooling2D(2, 2));
        model.AddLayer(new Conv2D(3, 16, 1, 0, new ReLU()));
        model.AddLayer(new MaxPooling2D(2, 2));
        model.AddLayer(new Dropout(0.2f));
        model.AddLayer(new Flatten());
        model.AddLayer(new Dense(128, new ReLU()));
        model.AddLayer(new Dense(10, new Softmax()));        
        model.Optimize(new Adam(), new CategoricalCrossEntropy());

        cout << model.Summary();

        Tensor input, output;
        LoadMnistData("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte", input, output, false, 6000);
        Tensor validationInput, validationOutput;
        LoadMnistData("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte", validationInput, validationOutput, false, 1000);

        model.Fit(input, output, 200, 10, &validationInput, &validationOutput, 2, ETrack::All);

        cout << model.TrainSummary();

        cin.get();
        return;
    }
};
