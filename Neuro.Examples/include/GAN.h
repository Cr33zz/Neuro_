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

        auto ganModel = new Sequential();
        ganModel->AddLayer(generator);
        ganModel->AddLayer(discriminator);
        ganModel->Optimize(new Adam(), new BinaryCrossEntropy());
        cout << ganModel->Summary();

        cin.get();
        return;
    }

private:
    static ModelBase* CreateGenerator(uint32_t inputsNum)
    {
        auto model = new Sequential("generator");
        model->AddLayer(new Dense(inputsNum, 256, new LeakyReLU(0.2f)));
        model->AddLayer(new Dense(512, new LeakyReLU(0.2f)));
        model->AddLayer(new Dense(1024, new LeakyReLU(0.2f)));
        model->AddLayer(new Dense(784, new Tanh()));
        model->Optimize(new Adam(), new BinaryCrossEntropy());
        cout << model->Summary();
        return model;
    }

    static ModelBase* CreateDiscriminator()
    {
        auto model = new Sequential("discriminator");
        model->AddLayer(new Dense(784, 1024, new LeakyReLU(0.2f)));
        model->AddLayer(new Dropout(0.3f));
        model->AddLayer(new Dense(512, new LeakyReLU(0.2f)));
        model->AddLayer(new Dropout(0.3f));
        model->AddLayer(new Dense(256, new LeakyReLU(0.2f)));
        model->AddLayer(new Dense(1, new Sigmoid()));
        model->Optimize(new Adam(), new BinaryCrossEntropy());
        cout << model->Summary();
        return model;
    }
};
