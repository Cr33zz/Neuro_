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

        const Shape IMG_SHAPE(128, 128, 3);
        const uint32_t PATCH = (uint32_t)(IMG_SHAPE.Height() / pow(2, 4));
        const Shape DISC_PATCH(PATCH, PATCH, 1);

        const uint32_t GF = 32;
        const uint32_t DF = 64;

        const float LAMBDA_CYCLE_WEIGHT = 10.f; // cycle consistency loss weight
        const float LAMBDA_ID_WEIGHT = 0.1f * LAMBDA_CYCLE_WEIGHT; // identity loss weight

        const int EPOCHS = 150;
        const uint32_t BATCH_SIZE = 1;
        const float LEARNING_RATE = 0.0002f;
        const float ADAM_BETA1 = 0.5f;

        cout << "Example: CycleGAN" << endl;

        const string NAME = "apples2oranges";
        auto trainFilesA = LoadFilesList("data/" + NAME + "/trainA", false, true);
        auto trainFilesB = LoadFilesList("data/" + NAME + "/trainB", false, true);
        
        // setup models
        auto gABModel = CreateGenerator(IMG_SHAPE, GF);
        auto gBAModel = CreateGenerator(IMG_SHAPE, GF);
        cout << "Generator AB" << endl << gABModel->Summary();
        cout << "Generator BA" << endl << gBAModel->Summary();
        auto dAModel = CreateDiscriminator(IMG_SHAPE, DF);
        auto dBModel = CreateDiscriminator(IMG_SHAPE, DF);
        dAModel->Optimize(new Adam(LEARNING_RATE, ADAM_BETA1), new MeanSquareError(), {}, All);
        dBModel->Optimize(new Adam(LEARNING_RATE, ADAM_BETA1), new MeanSquareError(), {}, All);
        cout << "Discriminator A" << endl << dAModel->Summary();
        cout << "Discriminator B" << endl << dBModel->Summary();

        //auto inSrc = (new Input(IMG_SHAPE))->Outputs()[0];
        auto inputImgA = new Placeholder(IMG_SHAPE);
        auto inputImgB = new Placeholder(IMG_SHAPE);

        // translate image A -> B
        auto fakeB = gABModel->Call(inputImgA)[0];
        // translate image B -> A
        auto fakeA = gBAModel->Call(inputImgB)[0];

        // translate fake image B back to A
        auto reconstructedA = gBAModel->Call(fakeB)[0];
        // translate fake image A back to B
        auto reconstructedB = gABModel->Call(fakeA)[0];

        // identity mapping
        auto imgAIdentity = gBAModel->Call(inputImgA)[0];
        auto imgBIdentity = gABModel->Call(inputImgB)[0];

        auto validA = dAModel->Call(fakeA)[0];
        auto validB = dBModel->Call(fakeB)[0];

        auto ganModel = Flow(
            { inputImgA, inputImgB }, 
            { validA, validB, reconstructedA, reconstructedB, imgAIdentity, imgBIdentity });
        ganModel.Optimize(
            new Adam(LEARNING_RATE, ADAM_BETA1), 
            { new MeanSquareError(), new MeanSquareError(), new MeanAbsoluteError(), new MeanAbsoluteError(), new MeanAbsoluteError(), new MeanAbsoluteError() }, 
            { 1.f, 1.f, LAMBDA_CYCLE_WEIGHT, LAMBDA_CYCLE_WEIGHT, LAMBDA_ID_WEIGHT, LAMBDA_ID_WEIGHT });

        // training
        auto fakeLabels = zeros(validA->GetShape());
        auto realLabels = ones(validA->GetShape());

        auto realImgA = zeros(inputImgA->GetShape());
        auto realImgB = zeros(inputImgB->GetShape());

        CustomMultiImageLoader loader(trainFilesA, trainFilesB, BATCH_SIZE);
        DataPreloader preloader({ &realImgA, &realImgB }, { &loader }, 5);
        
        const size_t STEPS = (size_t)((trainFilesA.size() + trainFilesB.size()) * 0.5f) * EPOCHS;

        Tqdm progress(STEPS, 0);
        progress.ShowEta(true).ShowElapsed(false).ShowPercent(false);
        for (size_t s = 0; s < STEPS; ++s, progress.NextStep())
        {
            //load next conditional and expected images
            preloader.Load();

            // optimize discriminator
            dAModel->SetTrainable(true);
            dBModel->SetTrainable(true);
            gABModel->SetTrainable(false);
            gBAModel->SetTrainable(false);
            
            auto fakeImgB = gABModel->Predict(realImgA)[0];
            auto fakeImgA = gBAModel->Predict(realImgB)[0];

            auto dAReal = dAModel->TrainOnBatch(realImgA, realLabels);
            auto dAFake = dAModel->TrainOnBatch(*fakeImgA, fakeLabels);
            float dALoss = 0.5f * (get<0>(dAReal) + get<0>(dAFake));

            auto dBReal = dBModel->TrainOnBatch(realImgB, realLabels);
            auto dBFake = dBModel->TrainOnBatch(*fakeImgB, fakeLabels);
            float dBLoss = 0.5f * (get<0>(dBReal) + get<0>(dBFake));

            float dLoss = 0.5f * (dALoss + dBLoss);

            // optimize generator
            dAModel->SetTrainable(false);
            dBModel->SetTrainable(false);
            gABModel->SetTrainable(true);
            gBAModel->SetTrainable(true);

            float ganLoss = get<0>(ganModel.TrainOnBatch({ &realImgA, &realImgB }, { &realLabels, &realLabels, &realImgA, &realImgB, &realImgA, &realImgB }));
            

            if (s % 50 == 0)
            {
                dAModel->SaveWeights(NAME + "_discA.h5");
                dBModel->SaveWeights(NAME + "_discB.h5");
                gABModel->SaveWeights(NAME + "_genAB.h5");
                gBAModel->SaveWeights(NAME + "_genBA.h5");
                Tensor tmp(Shape(IMG_SHAPE.Width() * 3, IMG_SHAPE.Height(), IMG_SHAPE.Depth(), BATCH_SIZE));
                Tensor::Concat(WidthAxis, { &realImgA, fakeImgB, &realImgB, fakeImgA }, tmp);
                tmp.Add(1.f).Mul(127.5f).SaveAsImage(NAME + "_s" + PadLeft(to_string(s), 4, '0') + ".jpg", false, 1);
            }

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
