#include <iostream>
#include <string>
#include <vector>

#include "Neuro.h"

using namespace std;
using namespace Neuro;

class GAN
{
public:
    static void Run()
    {
        Tensor::SetDefaultOpMode(EOpMode::GPU);

        auto generator = CreateGenerator(100);
        auto discriminator = CreateDiscriminator();

        auto model = Sequential();
        model.AddLayer(generator);
        model.AddLayer(discriminator);

        cin.get();
        return;
    }

private:
    static ModelBase* CreateGenerator(uint32_t inputsNum)
    {
        auto model = new Sequential("GAN_generator");
        model->AddLayer(new Dense(inputsNum, 256, new LeakyReLU(0.2f)));
        model->AddLayer(new Dense(512, new LeakyReLU(0.2f)));
        model->AddLayer(new Dense(1024, new LeakyReLU(0.2f)));
        model->AddLayer(new Dense(784, new Tanh()));

        cout << model->Summary();

        /*auto net = new NeuralNetwork(model, );
        net->Optimize(new Adam(), new BinaryCrossEntropy());*/
        return model;
    }

    static ModelBase* CreateDiscriminator()
    {
        auto model = new Sequential("GAN_discriminator");
        model->AddLayer(new Dense(784, 1024, new LeakyReLU(0.2f)));
        model->AddLayer(new Dropout(0.3f));
        model->AddLayer(new Dense(512, new LeakyReLU(0.2f)));
        model->AddLayer(new Dropout(0.3f));
        model->AddLayer(new Dense(256, new LeakyReLU(0.2f)));
        model->AddLayer(new Dense(1, new Sigmoid()));

        cout << model->Summary();

        /*auto net = new NeuralNetwork(model, "GAN_discriminator");
        net->Optimize(new Adam(), new BinaryCrossEntropy());*/
        return model;
    }
};
