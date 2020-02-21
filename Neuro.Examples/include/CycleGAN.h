#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <iomanip>

#include "Neuro.h"

using namespace std;
using namespace Neuro;

class CycleGAN
{
public:
    struct CustomMultiImageLoader : public MultiImageLoader
    {
        CustomMultiImageLoader(const vector<string>& filesA, const vector<string>& filesB, uint32_t batchSize) : MultiImageLoader({ filesA, filesB }, batchSize) {}

        virtual size_t operator()(vector<Tensor>& dest, size_t loadIdx) override
        {
            size_t loaded = __super::operator()(dest, loadIdx);
            
            for (size_t i = 0; i < loaded; ++i)
            {
                dest[i].Sub(127.5f, dest[i]);
                dest[i].Div(127.5f, dest[i]);
            }

            return loaded;
        }
    };

    void Run()
    {
        Tensor::SetForcedOpMode(GPU);
        //GlobalRngSeed(1337);

        const Shape IMG_SHAPE(256, 256, 3);

        const uint32_t GF = 32;
        const uint32_t DF = 64;

        const float CYCLE_WEIGHT = 10.f; // cycle consistency loss weight
        const float IDENTITY_WEIGHT = 0.1f * CYCLE_WEIGHT; // identity loss weight

        const int EPOCHS = 150;
        const uint32_t BATCH_SIZE = 1;
        const float LEARNING_RATE = 0.0002f;
        const float ADAM_BETA1 = 0.5f;

        cout << "Example: CycleGAN" << endl;

        const string NAME = "horse2zebra";
        auto trainFilesA = LoadFilesList("data/" + NAME + "/trainA", false, true);
        auto trainFilesB = LoadFilesList("data/" + NAME + "/trainB", false, true);
        
        // setup models
        auto gABModel = CreateGenerator(IMG_SHAPE, GF);
        auto gBAModel = CreateGenerator(IMG_SHAPE, GF);
        //cout << "Generator AB" << endl << gABModel->Summary();
        //cout << "Generator BA" << endl << gBAModel->Summary();
        auto dAModel = CreateDiscriminator(IMG_SHAPE, DF);
        auto dBModel = CreateDiscriminator(IMG_SHAPE, DF);
        dAModel->Optimize(new Adam(LEARNING_RATE, ADAM_BETA1), new MeanSquareError(), {}, All);
        dBModel->Optimize(new Adam(LEARNING_RATE, ADAM_BETA1), new MeanSquareError(), {}, All);
        //cout << "Discriminator A" << endl << dAModel->Summary();
        //cout << "Discriminator B" << endl << dBModel->Summary();

        auto inputA = (new Input(IMG_SHAPE))->Outputs()[0];
        auto inputB = (new Input(IMG_SHAPE))->Outputs()[0];

        // translate image A -> B
        auto fakeB = gABModel->Call(inputA)[0];
        // translate image B -> A
        auto fakeA = gBAModel->Call(inputB)[0];

        // translate fake image B back to A
        auto reconstructedA = gBAModel->Call(fakeB)[0];
        // translate fake image A back to B
        auto reconstructedB = gABModel->Call(fakeA)[0];

        // identity mapping
        auto identityA = gBAModel->Call(inputA)[0];
        auto identityB = gABModel->Call(inputB)[0];

        auto validA = dAModel->Call(fakeA)[0];
        auto validB = dBModel->Call(fakeB)[0];

        auto ganModel = Flow(
            { inputA, inputB }, 
            { validA, validB, reconstructedA, reconstructedB, identityA, identityB });

        ganModel.Optimize(
            new Adam(LEARNING_RATE, ADAM_BETA1), 
            { new MeanSquareError(), new MeanSquareError(), new MeanAbsoluteError(), new MeanAbsoluteError(), new MeanAbsoluteError(), new MeanAbsoluteError() }, 
            { 1.f, 1.f, CYCLE_WEIGHT, CYCLE_WEIGHT, IDENTITY_WEIGHT, IDENTITY_WEIGHT });

        // training
        auto fakeLabels = zeros(Shape::From(validA->GetShape(), BATCH_SIZE));
        auto realLabels = ones(Shape::From(validA->GetShape(), BATCH_SIZE));

        auto realImgA = zeros(Shape::From(inputA->GetShape(), BATCH_SIZE));
        auto realImgB = zeros(Shape::From(inputA->GetShape(), BATCH_SIZE));

        CustomMultiImageLoader loader(trainFilesA, trainFilesB, BATCH_SIZE);
        DataPreloader preloader({ &realImgA, &realImgB }, { &loader }, 5);

        auto testImgA = realImgA;
        auto testImgB = realImgB;
        
        const size_t STEPS = min(trainFilesA.size(), trainFilesB.size()) * EPOCHS;

        Tqdm progress(STEPS, 0);
        progress.ShowEta(true).ShowElapsed(false).ShowPercent(false);
        for (size_t s = 0; s < STEPS; ++s, progress.NextStep())
        {
            //load next conditional and expected images
            preloader.Load();

            if (s == 0)
            {
                testImgA = realImgA;
                testImgB = realImgB;
            }

            if (s % 100 == 0)
            {
                dAModel->SaveWeights(NAME + "_discA.h5");
                dBModel->SaveWeights(NAME + "_discB.h5");
                gABModel->SaveWeights(NAME + "_genAB.h5");
                gBAModel->SaveWeights(NAME + "_genBA.h5");
                Tensor output(Shape(IMG_SHAPE.Width() * 3, IMG_SHAPE.Height() * 2, IMG_SHAPE.Depth(), BATCH_SIZE));

                Tensor ab(Shape(IMG_SHAPE.Width() * 3, IMG_SHAPE.Height(), IMG_SHAPE.Depth(), BATCH_SIZE));
                auto testFakeB = *gABModel->Predict(testImgA)[0];
                auto testReconstructedA = *gBAModel->Predict(testFakeB)[0];
                Tensor::Concat(WidthAxis, { &testImgA, &testFakeB, &testReconstructedA }, ab);

                Tensor ba(Shape(IMG_SHAPE.Width() * 3, IMG_SHAPE.Height(), IMG_SHAPE.Depth(), BATCH_SIZE));
                auto testFakeA = *gABModel->Predict(testImgB)[0];
                auto testReconstructedB = *gBAModel->Predict(testFakeA)[0];
                Tensor::Concat(WidthAxis, { &testImgB, &testFakeA, &testReconstructedB }, ba);

                Tensor::Concat(HeightAxis, { &ab, &ba }, output);

                output.Add(1.f).Mul(127.5f).SaveAsImage(NAME + "_s" + PadLeft(to_string(s), 4, '0') + ".jpg", false, 1);
            }

            // optimize discriminators
            dAModel->SetTrainable(true);
            dBModel->SetTrainable(true);
            gABModel->SetTrainable(false);
            gBAModel->SetTrainable(false);
            
            auto fakeImgB = *gABModel->Predict(realImgA)[0];
            auto fakeImgA = *gBAModel->Predict(realImgB)[0];

            auto dAReal = dAModel->TrainOnBatch(realImgA, realLabels);
            auto dAFake = dAModel->TrainOnBatch(fakeImgA, fakeLabels);
            float dALoss = 0.5f * (get<0>(dAReal) + get<0>(dAFake));

            auto dBReal = dBModel->TrainOnBatch(realImgB, realLabels);
            auto dBFake = dBModel->TrainOnBatch(fakeImgB, fakeLabels);
            float dBLoss = 0.5f * (get<0>(dBReal) + get<0>(dBFake));

            float dLoss = 0.5f * (dALoss + dBLoss);

            // optimize generators
            dAModel->SetTrainable(false);
            dBModel->SetTrainable(false);
            gABModel->SetTrainable(true);
            gBAModel->SetTrainable(true);

            float ganLoss = get<0>(ganModel.TrainOnBatch({ &realImgA, &realImgB }, { &realLabels, &realLabels, &realImgA, &realImgB, &realImgA, &realImgB }));

            stringstream extString;
            extString << setprecision(4) << fixed << " - dLoss: " << dLoss << " - ganLoss: " << ganLoss;
            progress.SetExtraString(extString.str());
        }
    }

    //////////////////////////////////////////////////////////////////////////
    void RunDiscriminatorTrainTest()
    {
        //Tensor::SetDefaultOpMode(GPU);

        //GlobalRngSeed(1338);

        ///*Debug::LogAllGrads();
        //Debug::LogAllOutputs();*/

        //const Shape IMG_SHAPE(256, 256, 3);
        //const uint32_t PATCH_SIZE = 64;
        //const uint32_t BATCH_SIZE = 1;
        //const uint32_t STEPS = 150;
        //const float LEARNING_RATE = 0.0002f;
        //const float ADAM_BETA1 = 0.5f;

        //auto trainFiles = LoadFilesList("data/flowers", false, true);

        //Tensor condImages(Shape::From(IMG_SHAPE, BATCH_SIZE), "cond_image");
        //Tensor realImages(Shape::From(IMG_SHAPE, BATCH_SIZE), "output_image");

        //// setup models
        //auto gModel = CreateGenerator(IMG_SHAPE);
        ////cout << "Generator" << endl << gModel->Summary();
        ////auto dModel = CreatePatchDiscriminator(IMG_SHAPE, PATCH_SIZE, BATCH_SIZE > 1);
        //auto dModel = CreateDiscriminator(IMG_SHAPE);
        //dModel->Optimize(new Adam(LEARNING_RATE, ADAM_BETA1), new BinaryCrossEntropy(), {}, All);
        ////cout << "Discriminator" << endl << dModel->Summary();

        //auto inSrc = (new Input(IMG_SHAPE))->Outputs()[0];
        //auto genOut = gModel->Call(inSrc, "generator")[0];
        //auto disOut = dModel->Call({ inSrc, genOut }, "discriminator");

        //Tensor one(Shape(1, 1, 1, BATCH_SIZE)); one.One();

        //// labels consist of two values [fake_prob, real_prob]
        //Tensor fakeLabels(Shape::From(dModel->OutputShapesAt(-1)[0], BATCH_SIZE), "fake_labels"); fakeLabels.Zero();
        //one.FuseSubTensor2D(0, 0, fakeLabels); // generate [1, 0] batch
        //Tensor realLabels(Shape::From(dModel->OutputShapesAt(-1)[0], BATCH_SIZE), "real_lables"); realLabels.Zero();
        //one.FuseSubTensor2D(1, 0, realLabels);

        //// setup data preloader
        //EdgeImageLoader loader(trainFiles, BATCH_SIZE, 1, 1337);
        //DataPreloader preloader({ &condImages, &realImages }, { &loader }, 5);

        //for (uint32_t e = 1; e <= STEPS; ++e)
        //{
        //    preloader.Load();

        //    // generate fake images from condition
        //    //Tensor fakeImages = *gModel->Predict(condImages)[0];
        //    Tensor fakeImages(Shape::From(IMG_SHAPE, BATCH_SIZE));
        //    fakeImages.FillWithFunc([]() { return Uniform::NextSingle(-1, 1); });

        //    auto realTrainData = dModel->TrainOnBatch({ &realImages }, { &realLabels });
        //    //cout << get<0>(realTrainData) << endl;
        //    auto fakeTrainData = dModel->TrainOnBatch({ &fakeImages }, { &fakeLabels });
        //    //cout << get<0>(fakeTrainData) << endl;

        //    cout << ">" << e << setprecision(4) << fixed << " loss=" << (get<0>(realTrainData) + get<0>(fakeTrainData)) * 0.5f << " real=" << round(get<1>(realTrainData) * 100) << "% fake=" << round(get<1>(fakeTrainData) * 100) << "%" << endl;
        //}
    }

    ModelBase* CreateGenerator(const Shape& imgShape, uint32_t filtersStart);
    //ModelBase* CreatePatchDiscriminator(const Shape& imgShape, uint32_t patchSize, bool useMiniBatchDiscrimination = true);
    ModelBase* CreateDiscriminator(const Shape& imgShape, uint32_t filtersStart);
};
