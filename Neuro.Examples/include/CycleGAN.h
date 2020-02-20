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
    //// Splits source image into 2 images along width axis
    //struct SplitImageLoader : public ImageLoader
    //{
    //    SplitImageLoader(const vector<string>& files, uint32_t batchSize, uint32_t seed = 0) :ImageLoader(files, batchSize), m_Rng(seed) {}

    //    virtual size_t operator()(vector<Tensor>& dest, size_t loadIdx) override
    //    {
    //        auto& img1 = dest[loadIdx];
    //        auto& img2 = dest[loadIdx + 1];

    //        NEURO_ASSERT(img1.Width() == img2.Width(), "");
    //        NEURO_ASSERT(img1.Height() == img2.Height(), "");

    //        img1.ResizeBatch(m_BatchSize);
    //        img1.OverrideHost();
    //        img2.ResizeBatch(m_BatchSize);
    //        img2.OverrideHost();

    //        Tensor t1(Shape::From(img1.GetShape(), 1));
    //        Tensor t2(Shape::From(img2.GetShape(), 1));
    //        tensor_ptr_vec_t tmp{ &t1, &t2 };

    //        for (uint32_t n = 0; n < m_BatchSize; ++n)
    //        {
    //            const auto& file = m_Files[m_Rng.Next((int)m_Files.size())];
    //            auto img = LoadImage(file, img1.Width() * 2, img1.Height());
    //            img.Split(WidthAxis, tmp);

    //            t1.Sub(127.5f).Div(127.5f).CopyBatchTo(0, (uint32_t)n, img1);
    //            t2.Sub(127.5f).Div(127.5f).CopyBatchTo(0, (uint32_t)n, img2);
    //        }

    //        img1.CopyToDevice();
    //        img2.CopyToDevice();
    //        return 2;
    //    }

    //private:
    //    Random m_Rng;
    //};

    void Run()
    {
        Tensor::SetForcedOpMode(GPU);
        //GlobalRngSeed(1337);

        const Shape IMG_SHAPE(256, 256, 3);
        const uint32_t PATCH = (uint32_t)(IMG_SHAPE.Height() / pow(2, 4));
        const Shape DISC_PATCH(PATCH, PATCH, 1);

        const uint32_t GF = 32;
        const uint32_t DF = 64;

        const float LAMBDA_CYCLE_WEIGHT = 10.f; // cycle consistency loss weight
        const float LAMBDA_ID_WEIGHT = 0.1f * LAMBDA_CYCLE; // identity loss weight

        const int EPOCHS = 150;
        const uint32_t BATCH_SIZE = 1;
        const float LEARNING_RATE = 0.0002f;
        const float ADAM_BETA1 = 0.5f;

        cout << "Example: CycleGAN" << endl;

        const string NAME = "flowers";
        auto trainFiles = LoadFilesList("data/flowers", false, true);
        /*const string NAME = "facades";
        auto trainFiles = LoadFilesList("data/facades/train", false, true);*/

        // setup models
        auto gABModel = CreateGenerator(IMG_SHAPE);
        auto gBAModel = CreateGenerator(IMG_SHAPE);
        cout << "Generator AB" << endl << gABModel->Summary();
        cout << "Generator BA" << endl << gBAModel->Summary();
        auto dAModel = CreateDiscriminator(IMG_SHAPE);
        auto dBModel = CreateDiscriminator(IMG_SHAPE);
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
        auto fakeLabels = new Constant(zeros(validA->GetShape()), "fake_labels");
        auto realLabels = new Constant(ones(validA->GetShape()), "real_labels");

        DataPreloader preloader({ realImgA->OutputPtr(), realImgB->OutputPtr() }, { &loader }, 5);
        
        const size_t STEPS = trainFiles.size() * EPOCHS;

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

            dAModel->TrainOnBatch(realImgA, realLabels);
            dAModel->TrainOnBatch(fakeImgA, fakeLabels);

            // optimize generator
            dAModel->SetTrainable(false);
            dBModel->SetTrainable(false);
            gABModel->SetTrainable(true);
            gBAModel->SetTrainable(true);

            ganModel.TrainOnBatch({ realImgA, realImgB }, { realLabels, realLabels, realImgA, realImgB, realImgA, realImgB });
            

            if (s % 50 == 0)
            {
                dModel->SaveWeights(NAME + "_disc.h5");
                gModel->SaveWeights(NAME + "_gen.h5");
                Tensor tmp(Shape(IMG_SHAPE.Width() * 3, IMG_SHAPE.Height(), IMG_SHAPE.Depth(), BATCH_SIZE));
                Tensor::Concat(WidthAxis, { inputImg->OutputPtr(), &_genImg, targetImg->OutputPtr() }, tmp);
                tmp.Add(1.f).Mul(127.5f).SaveAsImage(NAME + "_s" + PadLeft(to_string(s), 4, '0') + ".jpg", false, 1);
            }

            stringstream extString;
            extString << setprecision(4) << fixed << " - dLoss: " << dLoss << " - ganLoss: " << _ganLoss << " - l1Loss: " << _l1Loss << " - genLoss: " << _genLoss;
            progress.SetExtraString(extString.str());
        }

        //ganModel->SaveWeights(NAME + ".h5");
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

    ModelBase* CreateGenerator(const Shape& imgShape);
    //ModelBase* CreatePatchDiscriminator(const Shape& imgShape, uint32_t patchSize, bool useMiniBatchDiscrimination = true);
    ModelBase* CreateDiscriminator(const Shape& imgShape);
};
