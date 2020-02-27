#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <iomanip>

#include "Neuro.h"

using namespace std;
using namespace Neuro;

class SRGAN
{
public:
    void Run()
    {
        Tensor::SetForcedOpMode(GPU);
        //GlobalRngSeed(1337);

        const uint32_t LOW_RES_DIM = 64;

        const Shape LR_IMG_SHAPE(LOW_RES_DIM, LOW_RES_DIM, 3);
        const Shape HR_IMG_SHAPE(LOW_RES_DIM * 4, LOW_RES_DIM * 4, 3);

        const size_t RESIDUAL_BLOCKS_NUM = 16;

        const uint32_t GF = 64;
        const uint32_t DF = 64;

        const int EPOCHS = 150;
        const uint32_t BATCH_SIZE = 1;
        const float LEARNING_RATE = 0.0002f;
        const float ADAM_BETA1 = 0.5f;

        cout << "Example: SRGAN" << endl;

        const string NAME = "horse2zebra";
        auto trainFiles = LoadFilesList("data/" + NAME, false, true);

        // setup models
        auto gen = CreateGenerator(LR_IMG_SHAPE, HR_IMG_SHAPE, GF, RESIDUAL_BLOCKS_NUM);
        //cout << "Generator" << endl << gen->Summary();
        auto disc = CreateDiscriminator(HR_IMG_SHAPE, DF);
        disc->Optimize(new Adam(LEARNING_RATE, ADAM_BETA1), new MeanSquareError(), {}, All);
        //cout << "Discriminator" << endl << disc->Summary();

        auto imgLr = (new Input(LR_IMG_SHAPE))->Outputs()[0];
        auto imgHr = (new Input(HR_IMG_SHAPE))->Outputs()[0];

        // generate high resolution image
        auto fakeHr = gen->Call(imgLr)[0];

        auto vgg = VGG19::CreateModel(NCHW, HR_IMG_SHAPE, false, MaxPool, "data/");
        vgg->SetTrainable(false);

        auto model = Flow(vgg->InputsAt(-1), vgg->Layer("block3_conv2")->Outputs());

        auto fakeFeatures = model(fakeHr)[0];
        auto validity = disc->Call(fakeHr)[0];

        auto ganModel = Flow(vector<TensorLike*>{ imgLr, imgHr }, { validity, fakeFeatures });
        ganModel.Optimize(new Adam(LEARNING_RATE, ADAM_BETA1), { new BinaryCrossEntropy(), new MeanSquareError() }, { 1e-3f, 1.f });

        // training
        auto fakeLabels = zeros(Shape::From(validity->GetShape(), BATCH_SIZE));
        auto realLabels = ones(Shape::From(validity->GetShape(), BATCH_SIZE));

        auto realLrImg = zeros(Shape::From(LR_IMG_SHAPE, BATCH_SIZE));
        auto realHrImg = zeros(Shape::From(HR_IMG_SHAPE, BATCH_SIZE));

        CustomMultiImageLoader loader(trainFiles, BATCH_SIZE);
        DataPreloader preloader({ &realLrImg, &realHrImg }, { &loader }, 5);

        auto testLrImg = realLrImg;
        auto testHrImg = realHrImg;

        const size_t STEPS = trainFiles.size() * EPOCHS;

        Tqdm progress(STEPS, 0);
        progress.ShowEta(true).ShowElapsed(false).ShowPercent(false);
        for (size_t s = 0; s < STEPS; ++s, progress.NextStep())
        {
            //load next conditional and expected images
            preloader.Load();

            if (s == 0)
            {
                testLrImg = realLrImg;
                testHrImg = realHrImg;
            }

            if (s % 100 == 0)
            {
                disc->SaveWeights(NAME + "_disc.h5");
                gen->SaveWeights(NAME + "_gen.h5");
                Tensor output(Shape(HR_IMG_SHAPE.Width() * 3, HR_IMG_SHAPE.Height(), HR_IMG_SHAPE.Depth(), BATCH_SIZE));

                auto testFakeHrImg = *gen->Predict(testLrImg)[0];
                Tensor::Concat(WidthAxis, { &realLrImg.UpSample2D(2), &testFakeHrImg, &realHrImg }, output);
                output.Add(1.f).Mul(127.5f).SaveAsImage(NAME + "_s" + PadLeft(to_string(s), 4, '0') + ".jpg", false, 1);
            }

            // optimize discriminators
            disc->SetTrainable(true);
            gen->SetTrainable(false);

            auto fakeHrImg = *gen->Predict(realLrImg)[0];

            auto dReal = disc->TrainOnBatch(realHrImg, realLabels);
            auto dFake = disc->TrainOnBatch(fakeHrImg, fakeLabels);
            float dLoss = 0.5f * (get<0>(dReal) + get<0>(dFake));
            
            preloader.Load();

            // optimize generators
            disc->SetTrainable(false);
            gen->SetTrainable(true);

            auto realFeat = *model.Predict(realHrImg)[0];

            float ganLoss = get<0>(ganModel.TrainOnBatch({ &realLrImg, &realHrImg }, { &realLabels, &realFeat }));

            stringstream extString;
            extString << setprecision(4) << fixed << " - dLoss: " << dLoss << " - ganLoss: " << ganLoss;
            progress.SetExtraString(extString.str());
        }
    }

    ModelBase* CreateGenerator(const Shape& lrShape, const Shape& hrShape, uint32_t filtersStart, size_t residualBlocksNum);
    ModelBase* CreateDiscriminator(const Shape& hrShape, uint32_t filtersStart);
};
