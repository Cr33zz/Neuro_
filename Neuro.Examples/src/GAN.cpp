#include "GAN.h"

//////////////////////////////////////////////////////////////////////////
void GAN::Run()
{
    Tensor::SetDefaultOpMode(GPU);
    GlobalRngSeed(1337);

    //Debug::LogAllGrads();
    //Debug::LogAllOutputs();
    //Debug::LogOutput("loss/cross_entropy/negative");

    cout << "Example: " << Name() << endl;

    auto gModel = CreateGenerator(100);
    //cout << "Generator" << endl << gModel->Summary();
    auto dModel = CreateDiscriminator();
    //cout << "Discriminator" << endl << dModel->Summary();

    auto ganModel = new Sequential(Name());
    ganModel->AddLayer(gModel);
    ganModel->AddLayer(dModel);
    ganModel->Optimize(new Adam(0.0002f, 0.5f), new BinaryCrossEntropy());
    //cout << "GAN" << endl << ganModel->Summary();

    Tensor images;
    LoadImages(images);
    m_ImageShape = Shape(images.Width(), images.Height(), images.Depth());
    images.Map([](float x) { return (x - 127.5f) / 127.5f; }, images);
    images.Reshape(Shape::From(dModel->InputShapesAt(-1)[0], images.Batch()));
    images.Name("images");

    const uint32_t BATCH_SIZE = 128;
    const uint32_t BATCHES_PER_EPOCH = images.Batch() / BATCH_SIZE;
    const uint32_t EPOCHS = 100;

    Tensor testNoise(Shape::From(gModel->InputShapesAt(-1)[0], 100), "test_noise"); testNoise.FillWithFunc([]() { return Normal::NextSingle(0, 1); });
    
    Tensor noise(Shape::From(gModel->InputShapesAt(-1)[0], BATCH_SIZE), "noise");
    Tensor real(Shape::From(dModel->OutputShapesAt(-1)[0], BATCH_SIZE), "real"); real.FillWithValue(1.f);
    
    Tensor noiseHalf(Shape::From(gModel->InputShapesAt(-1)[0], BATCH_SIZE / 2), "noise_half");
    Tensor realHalf(Shape::From(dModel->OutputShapesAt(-1)[0], BATCH_SIZE / 2), "real_half"); realHalf.FillWithValue(1.f);
    Tensor fakeHalf(Shape::From(dModel->OutputShapesAt(-1)[0], BATCH_SIZE / 2), "fake_half"); fakeHalf.FillWithValue(0.f);

    for (uint32_t e = 1; e <= EPOCHS; ++e)
    {
        cout << "Epoch " << e << endl;

        float totalGanLoss = 0.f;
        
        Tqdm progress(BATCHES_PER_EPOCH, 0);
        progress.ShowEta(true).ShowElapsed(false).ShowPercent(false);
        for (uint32_t i = 0; i < BATCHES_PER_EPOCH; ++i, progress.NextStep())
        {
            noiseHalf.FillWithFunc([]() { return Normal::NextSingle(0, 1); });
            
            // generate fake images from noise
            Tensor fakeImages = *gModel->Predict(noiseHalf)[0];
            fakeImages.Name("fake_images");
            // grab random batch of real images
            Tensor realImages = images.GetRandomBatches(BATCH_SIZE / 2);
            realImages.Name("real_images");

            // perform step of training discriminator to distinguish fake from real images
            dModel->SetTrainable(true);
            float dRealLoss = get<0>(dModel->TrainOnBatch(realImages, realHalf));
            float dFakeLoss = get<0>(dModel->TrainOnBatch(fakeImages, fakeHalf));

            noise.FillWithFunc([]() { return Normal::NextSingle(0, 1); });

            // perform step of training generator to generate more real images (the more discriminator is confident that a particular image is fake the more generator will learn)
            dModel->SetTrainable(false);
            float ganLoss = get<0>(ganModel->TrainOnBatch(noise, real));

            stringstream extString;
            extString << setprecision(4) << fixed << " - real_l: " << dRealLoss << " - fake_l: " << dFakeLoss << " - gan_l: " << ganLoss;
            progress.SetExtraString(extString.str());

            if (i % 50 == 0)
                gModel->Predict(testNoise)[0]->Map([](float x) { return x * 127.5f + 127.5f; }).Reshaped(Shape(m_ImageShape.Width(), m_ImageShape.Height(), m_ImageShape.Depth(), -1)).SaveAsImage(Name() + "_e" + PadLeft(to_string(e), 4, '0') + "_b" + PadLeft(to_string(i), 4, '0') + ".png", false);
        }
    }

    ganModel->SaveWeights(Name() + ".h5");

    cin.get();
}

//////////////////////////////////////////////////////////////////////////
void GAN::RunDiscriminatorTrainTest()
{
    Tensor::SetDefaultOpMode(GPU);

    //GlobalRngSeed(1337);

    auto dModel = CreateDiscriminator();
    cout << dModel->Summary();

    Tensor images;
    LoadImages(images);
    images.Map([](float x) { return x / 127.5f - 1.f; }, images);
    images.Reshape(Shape::From(dModel->InputShapesAt(-1)[0], images.Batch()));

    const uint32_t BATCH_SIZE = 32;
    const uint32_t EPOCHS = 25;

    Tensor real(Shape::From(dModel->OutputShapesAt(-1)[0], BATCH_SIZE)); real.FillWithValue(1.f);
    Tensor fake(Shape::From(dModel->OutputShapesAt(-1)[0], BATCH_SIZE)); fake.FillWithValue(0.f);

    for (uint32_t e = 1; e <= EPOCHS; ++e)
    {
        Tensor fakeImages(Shape::From(dModel->InputShapesAt(-1)[0], BATCH_SIZE)); fakeImages.FillWithFunc([]() { return Uniform::NextSingle(-1, 1); });
        Tensor realImages = images.GetRandomBatches(BATCH_SIZE);

        auto realTrainData = dModel->TrainOnBatch(realImages, real);
        auto fakeTrainData = dModel->TrainOnBatch(fakeImages, fake);

        cout << ">" << e << setprecision(4) << fixed << " loss=" << (get<0>(realTrainData) + get<0>(fakeTrainData)) * 0.5f << " real=" << round(get<1>(realTrainData)*100) << "% fake=" << round(get<1>(fakeTrainData)*100) << "%" << endl;
    }

    cin.get();
}

//////////////////////////////////////////////////////////////////////////
void GAN::LoadImages(Tensor& images)
{
    Tensor labels;
    LoadMnistData("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte", images, labels, false, false, -1);
}

//////////////////////////////////////////////////////////////////////////
ModelBase* GAN::CreateGenerator(uint32_t inputsNum)
{
    auto model = new Sequential("generator");
    model->AddLayer(new Dense(inputsNum, 256, new LeakyReLU(0.2f)));
    model->AddLayer(new Dense(512, new LeakyReLU(0.2f)));
    model->AddLayer(new Dense(1024, new LeakyReLU(0.2f)));
    model->AddLayer(new Dense(784, new Tanh()));
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
    return model;
}

