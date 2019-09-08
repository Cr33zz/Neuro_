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

        auto model = new Sequential("mnist_conv");
        model->AddLayer(new Conv2D(Shape(28, 28, 1), 3, 32, 1, 0, new ReLU()));
        model->AddLayer(new MaxPooling2D(model->LastLayer(), 2));
        model->AddLayer(new Dropout(model->LastLayer(), 0.2f));
        model->AddLayer(new Flatten(model->LastLayer()));
        model->AddLayer(new Dense(model->LastLayer(), 128, new ReLU()));
        model->AddLayer(new Dense(model->LastLayer(), 10, new Softmax()));        

        cout << model->Summary();

        model->Optimize(new Adam(), new CategoricalCrossEntropy());

        Tensor input, output;
        LoadMnistData("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte", input, output, false, 6000);
        Tensor validationInput, validationOutput;
        LoadMnistData("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte", validationInput, validationOutput, false, 1000);

        model->Fit(input, output, 200, 10, &validationInput, &validationOutput, 2, ETrack::All);

        cout << model->TrainSummary();

        cin.get();
        return;
    }
};
