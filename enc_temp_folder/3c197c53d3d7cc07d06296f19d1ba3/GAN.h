#include <iostream>
#include <string>
#include <vector>
#include <iomanip>

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
        ganModel->Optimize(new Adam(0.0002f, 0.5f), new BinaryCrossEntropy());
        cout << ganModel->Summary();

        Tensor images, labels;
        LoadMnistData("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte", images, labels, false, false, 60000);
        images.Map([](float x) { return (x - 127.5f) / 127.5f; }, images);
        images.Reshape(Shape(1, 784, 1, -1));

        const uint32_t BATCH_SIZE = 64;
        const uint32_t HALF_BATCH_SIZE = BATCH_SIZE / 2;
        const uint32_t EPOCHS = 100;

        uint32_t batchCount = images.Batch() / BATCH_SIZE;
        Tensor discriminatorFakeInput(Shape::From(discriminator->InputShape(), HALF_BATCH_SIZE));
        Tensor discriminatorRealInput(Shape::From(discriminator->InputShape(), HALF_BATCH_SIZE));
        Tensor discriminatorFakeOutput(Shape::From(discriminator->OutputShape(), HALF_BATCH_SIZE));
        Tensor discriminatorRealOutput(Shape::From(discriminator->OutputShape(), HALF_BATCH_SIZE));
        discriminatorFakeOutput.FillWithValue(1.f);
        discriminatorRealOutput.FillWithValue(0.f);
        Tensor ganHalfInput(Shape::From(generator->InputShape(), HALF_BATCH_SIZE));
        Tensor ganInput(Shape::From(generator->InputShape(), BATCH_SIZE));
        Tensor ganOutput(Shape::From(discriminator->OutputShape(), BATCH_SIZE));
        ganOutput.FillWithValue(0.f);
        
        for (uint32_t e = 1; e <= EPOCHS; ++e)
        {
            cout << "Epoch " << e << endl;

            float totalGanError = 0.f;
            float totalDiscriminatorError = 0.f;

            Tqdm progress(batchCount);
            for (uint32_t i = 0; i < batchCount; ++i, progress.NextStep())
            {
                ganHalfInput.FillWithFunc([]() { return Normal::NextSingle(0, 1); });

                Tensor generatedImages = generator->Predict(ganHalfInput)[0];
                Tensor realImages = images.GetRandomBatches(HALF_BATCH_SIZE);

                discriminator->SetTrainable(true);
                float discriminatorError = 0;
                discriminatorError += discriminator->TrainOnBatch(generatedImages, discriminatorFakeOutput);
                discriminatorError += discriminator->TrainOnBatch(realImages, discriminatorRealOutput);

                totalDiscriminatorError += discriminatorError * 0.5f;

                ganInput.FillWithFunc([]() { return Normal::NextSingle(0, 1); });

                discriminator->SetTrainable(false);
                totalGanError += ganModel->TrainOnBatch(ganInput, ganOutput);
            }

            cout << " - d_error: " << setprecision(4) << totalDiscriminatorError / BATCH_SIZE << " - g_error: " << totalGanError / BATCH_SIZE << endl;

            if (e == 1 || e % 10 == 0)
                generator->Output().Map([](float x) { return x * 127.5f + 127.5f; }).Reshaped(Shape(28, 28, 1, -1)).SaveAsImage("generator_" + to_string(e) + ".png", true);
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
        //cout << model->Summary();
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
        model->Optimize(new Adam(0.0002f, 0.5f), new BinaryCrossEntropy());
        //cout << model->Summary();
        return model;
    }
};
