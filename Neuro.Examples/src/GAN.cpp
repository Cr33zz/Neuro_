#include "GAN.h"

//////////////////////////////////////////////////////////////////////////
void GAN::Run()
{
    Tensor::SetDefaultOpMode(EOpMode::GPU);

    auto generator = CreateGenerator(100);
    auto discriminator = CreateDiscriminator();

    auto ganModel = new Sequential(Name());
    ganModel->AddLayer(generator);
    ganModel->AddLayer(discriminator);
    ganModel->Optimize(new Adam(0.0002f, 0.5f), new BinaryCrossEntropy());
    cout << ganModel->Summary();

    Tensor images, labels;
    LoadMnistData("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte", images, labels, false, false, 60000);
    images.Map([](float x) { return (x - 127.5f) / 127.5f; }, images);
    images.Reshape(Shape(1, 784, 1, -1));

    const uint32_t BATCH_SIZE = 64;
    const uint32_t EPOCHS = 100;

    uint32_t BATCHES_NUM = images.Batch() / BATCH_SIZE;
    
    Tensor noise(Shape::From(generator->InputShape(), BATCH_SIZE));
    Tensor real(Shape::From(discriminator->OutputShape(), BATCH_SIZE));
    real.FillWithValue(0.f);
    Tensor fake(Shape::From(discriminator->OutputShape(), BATCH_SIZE));
    fake.FillWithValue(1.f);

    for (uint32_t e = 1; e <= EPOCHS; ++e)
    {
        cout << "Epoch " << e << endl;

        float totalGanError = 0.f;
        float totalDiscriminatorError = 0.f;

        Tqdm progress(BATCHES_NUM);
        for (uint32_t i = 0; i < BATCHES_NUM; ++i, progress.NextStep())
        {
            noise.FillWithFunc([]() { return Normal::NextSingle(0, 1); });
            
            //generator->ForceLearningPhase(true); // without it batch normalization will not normalize in the first pass
            // generate fake images from noise
            Tensor genImages = generator->Predict(noise)[0];
            //generatedImages.Map([](float x) { return x * 127.5f + 127.5f; }).Reshaped(Shape(28, 28, 1, -1)).SaveAsImage("generated_images.png", false);
            // grab random batch of real images
            Tensor realImages = images.GetRandomBatches(BATCH_SIZE);

            // perform step of training discriminator to distinguish fake from real images
            discriminator->SetTrainable(true);
            float discriminatorError = 0;
            discriminatorError += discriminator->TrainOnBatch(genImages, fake);
            discriminatorError += discriminator->TrainOnBatch(realImages, real);

            totalDiscriminatorError += discriminatorError * 0.5f;

            // perform step of training generator to generate more real images (the more discriminator is confident that a particular image is fake the more generator will learn)
            discriminator->SetTrainable(false);
            totalGanError += ganModel->TrainOnBatch(noise, real);
        }

        cout << " - discriminator_error: " << setprecision(4) << totalDiscriminatorError / BATCHES_NUM << " - gan_error: " << totalGanError / BATCHES_NUM << endl;

        if (e == 1 || e % 10 == 0)
            generator->Output().Map([](float x) { return x * 127.5f + 127.5f; }).Reshaped(Shape(28, 28, 1, -1)).SaveAsImage(Name() + "_" + to_string(e) + ".png", false);
    }

    cin.get();
    return;
}

//////////////////////////////////////////////////////////////////////////
ModelBase* GAN::CreateGenerator(uint32_t inputsNum)
{
    auto model = new Sequential("generator");
    model->AddLayer(new Dense(inputsNum, 256, new LeakyReLU(0.2f)));
    model->AddLayer(new Dense(512, new LeakyReLU(0.2f)));
    model->AddLayer(new Dense(1024, new LeakyReLU(0.2f)));
    model->AddLayer(new Dense(784, new Tanh()));
    //cout << model->Summary();
    return model;
}

//////////////////////////////////////////////////////////////////////////
ModelBase* GAN::CreateDiscriminator()
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

