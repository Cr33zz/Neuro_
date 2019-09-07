#include <iostream>
#include <string>
#include <vector>

#include "Neuro.h"

using namespace std;
using namespace Neuro;

class DeepConvGAN
{
public:
    static void Run()
    {
        Tensor::SetDefaultOpMode(EOpMode::GPU);

        cin.get();
        return;
    }

private:
    static NeuralNetwork* CreateGenerator(uint32_t inputsNum)
    {
        auto model = new Sequential();
        model->AddLayer(new Dense(inputsNum, 127*7*7, new ReLU()));
        model->AddLayer(new Reshape(model->LastLayer(), Shape(7, 7, 128)));
        model->AddLayer(new Conv2D(model->LastLayer(), 3, 128, 1, 1, new ELU(1)));
        model->AddLayer(new Flatten(model->LastLayer()));
        model->AddLayer(new Dense(model->LastLayer(), 512, new ELU(1)));
        model->AddLayer(new Dense(model->LastLayer(), 3, new Softmax()));

        cout << model->Summary();

        auto net = new NeuralNetwork(model, "DCGAN");
        net->Optimize(new Adam(), new BinaryCrossEntropy());
        return net;
    }

    static NeuralNetwork* CreateDiscriminator()
    {

    }
};
