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

        Tensor images, labels;
        LoadMnistData("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte", images, labels, false, 6000);
        images.Reshape(Shape(1, 784, 1, -1));

        const uint32_t BATCH_SIZE = 128;
        const uint32_t EPOCHS = 40;

        uint32_t batchCount = images.Batch() / BATCH_SIZE;
        Tensor discriminatorInput(Shape::From(discriminator->InputShape(), 2 * BATCH_SIZE));
        Tensor discriminatorOutput(Shape::From(discriminator->OutputShape(), 2 * BATCH_SIZE));
        Tensor ganInput(Shape::From(generator->InputShape(), BATCH_SIZE));
        Tensor ganOutput(Shape::From(discriminator->OutputShape(), BATCH_SIZE));
        
        for (uint32_t e = 1; e <= EPOCHS; ++e)
        {
            cout << "Epoch " << e << endl;

            Tqdm progress(batchCount);
            for (uint32_t i = 0; i < batchCount; ++i, progress.NextStep())
            {
                ganInput.FillWithFunc([]() { return Normal::NextSingle(0, 1); });

                Tensor generatedImages = generator->Predict(ganInput)[0];
                Tensor realImages = images.GetRandomBatches(BATCH_SIZE);

                generatedImages.Concat(EAxis::Global, { &generatedImages, &realImages }, discriminatorInput);
                discriminatorOutput.Zero();
                discriminatorOutput.FillWithValue(0.9f, BATCH_SIZE);

                discriminator->SetTrainable(true);
                discriminator->TrainOnBatch(discriminatorInput, discriminatorOutput);

                ganInput.FillWithFunc([]() { return Normal::NextSingle(0, 1); });
                ganOutput.FillWithValue(1.f);

                discriminator->SetTrainable(false);
                ganModel->TrainOnBatch(ganInput, ganOutput);
            }

            if (e == 1 || e % 2 == 0)
                generator->Output().Reshaped(Shape(28, 28, 1, -1)).SaveAsImage("generator_" + to_string(e) + ".png", true);
        }

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
