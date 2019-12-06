#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <iomanip>

#include "Neuro.h"

using namespace std;
using namespace Neuro;

class Pix2Pix
{
public:
    struct Pix2PixImageLoader : public ImageLoader
    {
        Pix2PixImageLoader(const vector<string>& files, uint32_t batchSize, uint32_t upScaleFactor = 1, uint32_t seed = 0) :ImageLoader(files, batchSize, upScaleFactor), m_Rng(seed) {}

        virtual size_t operator()(vector<Tensor>& dest, size_t loadIdx) override
        {
            auto& cndImg = dest[loadIdx];
            auto& outImg = dest[loadIdx + 1];
            cndImg.ResizeBatch(m_BatchSize);
            cndImg.OverrideHost();
            outImg.ResizeBatch(m_BatchSize);
            outImg.OverrideHost();

            for (uint32_t n = 0; n < m_BatchSize; ++n)
            {
                const auto& file = m_Files[m_Rng.Next((int)m_Files.size())];
                //cout << "Load: " << file << endl;
                auto img = LoadImage(file, cndImg.Width() * m_UpScaleFactor, cndImg.Height() * m_UpScaleFactor, cndImg.Width(), cndImg.Height());
                auto edges = CannyEdgeDetection(img).ToRGB();
                
                edges.Sub(127.5f).Div(127.5f).CopyBatchTo(0, (uint32_t)n, cndImg);
                img.Sub(127.5f).Div(127.5f).CopyBatchTo(0, (uint32_t)n, outImg);                
            }

            cndImg.CopyToDevice();
            outImg.CopyToDevice();
            return 2;
        }

    private:
        Random m_Rng;
    };

    void Run()
    {
        Tensor::SetForcedOpMode(GPU);
        //GlobalRngSeed(1337);

        const Shape IMG_SHAPE(256, 256, 3);
        const uint32_t PATCH_SIZE = 64;
        const uint32_t BATCH_SIZE = 1;
        const uint32_t STEPS = 100000;
        //const uint32_t STEPS = 6;

        cout << "Example: Pix2Pix" << endl;

        //auto trainFiles = LoadFilesList("e:/Downloads/flowers", false, true);
        auto trainFiles = LoadFilesList("f:/!TrainingData/flowers", false, true);

        Tensor condImages(Shape::From(IMG_SHAPE, BATCH_SIZE), "cond_image");
        Tensor realImages(Shape::From(IMG_SHAPE, BATCH_SIZE), "output_image");

        // setup data preloader
        Pix2PixImageLoader loader(trainFiles, BATCH_SIZE, 1);
        DataPreloader preloader({ &condImages, &realImages }, { &loader }, 5);

        // setup models
        auto gModel = CreateGenerator(IMG_SHAPE);
        cout << "Generator" << endl << gModel->Summary();
        auto dModel = CreatePatchDiscriminator(IMG_SHAPE, PATCH_SIZE, BATCH_SIZE > 1);
        //cout << "Discriminator" << endl << dModel->Summary();

        auto inSrc = new Input(IMG_SHAPE);
        auto genOut = gModel->Call(inSrc->Outputs(), "generator");
        auto disOut = dModel->Call(genOut[0], "discriminator");

        auto ganModel = new Flow(inSrc->Outputs(), { disOut[0], genOut[0] }, "pix2pix");
        ganModel->Optimize(new Adam(0.0001f), { new BinaryCrossEntropy(), new MeanAbsoluteError() }, { 1.f, 100.f });
        ganModel->LoadWeights("pix2pix.h5", false, true);

        Tensor one(Shape(1, BATCH_SIZE)); one.One();

        // labels consist of two values [fake_prob, real_prob]
        Tensor fakeLabels(Shape::From(dModel->OutputShapesAt(-1)[0], BATCH_SIZE), "fake_labels"); fakeLabels.Zero();
        one.FuseSubTensor2D(0, 0, fakeLabels); // generate [1, 0] batch
        Tensor realLabels(Shape::From(dModel->OutputShapesAt(-1)[0], BATCH_SIZE), "real_lables"); realLabels.Zero();
        one.FuseSubTensor2D(1, 0, realLabels);

        size_t discriminatorTrainingSource = 1;

        Tqdm progress(STEPS, 0);
        progress.ShowEta(true).ShowElapsed(false).ShowPercent(false);
        for (uint32_t i = 0; i < STEPS; ++i, progress.NextStep())
        {
            //load next conditional and expected images
            preloader.Load();

            // perform step of training discriminator to distinguish fake from real images
            dModel->SetTrainable(true);

            float dLoss;
            if (discriminatorTrainingSource & 1)
            {
                dLoss = get<0>(dModel->TrainOnBatch({ &realImages }, { &realLabels }));
            }
            else
            {
                // generate fake images from condition
                Tensor fakeImages = *gModel->Predict(condImages)[0];
                dLoss = get<0>(dModel->TrainOnBatch({ &fakeImages }, { &fakeLabels }));

                if (discriminatorTrainingSource % 10 == 0)
                {
                    ganModel->SaveWeights("pix2pix.h5");
                    Tensor tmp(Shape(IMG_SHAPE.Width() * 3, IMG_SHAPE.Height(), IMG_SHAPE.Depth(), BATCH_SIZE));
                    Tensor::Concat(WidthAxis, { &condImages, &fakeImages, &realImages }, tmp);
                    tmp.Add(1.f).Mul(127.5f).SaveAsImage("pix2pix_s" + PadLeft(to_string(i), 4, '0') + ".jpg", false);
                }
            }
            ++discriminatorTrainingSource;
            
            // perform step of training generator to generate more real images
            dModel->SetTrainable(false);
            float ganLoss = get<0>(ganModel->TrainOnBatch({ &condImages }, { &realLabels, &realImages }));

            stringstream extString;
            extString << setprecision(4) << fixed << " - disc_l: " << dLoss << " - gan_l: " << ganLoss;
            progress.SetExtraString(extString.str());
        }

        ganModel->SaveWeights("pix2pix.h5");
    }

    //////////////////////////////////////////////////////////////////////////
    void RunDiscriminatorTrainTest()
    {
        Tensor::SetDefaultOpMode(GPU);

        GlobalRngSeed(1338);

        //Debug::LogAllGrads();
        //Debug::LogAllOutputs();

        const Shape IMG_SHAPE(256, 256, 3);
        const uint32_t PATCH_SIZE = 256;
        const uint32_t BATCH_SIZE = 4;
        const uint32_t EPOCHS = 150;

        //auto trainFiles = LoadFilesList("f:/!TrainingData/flowers", false, true);
        auto trainFiles = LoadFilesList("e:/Downloads/flowers", false, true);

        Tensor condImages(Shape::From(IMG_SHAPE, BATCH_SIZE), "cond_image");
        Tensor realImages(Shape::From(IMG_SHAPE, BATCH_SIZE), "output_image");

        // setup models
        auto gModel = CreateGenerator(IMG_SHAPE);
        //cout << "Generator" << endl << gModel->Summary();
        auto dModel = CreatePatchDiscriminator(IMG_SHAPE, PATCH_SIZE, BATCH_SIZE > 1);
        //cout << "Discriminator" << endl << dModel->Summary();

        auto inSrc = new Input(IMG_SHAPE);
        auto genOut = gModel->Call(inSrc->Outputs(), "generator");
        auto disOut = dModel->Call(genOut[0], "discriminator");

        Tensor one(Shape(1, BATCH_SIZE)); one.One();

        // labels consist of two values [fake_prob, real_prob]
        Tensor fakeLabels(Shape::From(dModel->OutputShapesAt(-1)[0], BATCH_SIZE), "fake_labels"); fakeLabels.Zero();
        one.FuseSubTensor2D(0, 0, fakeLabels); // generate [1, 0] batch
        Tensor realLabels(Shape::From(dModel->OutputShapesAt(-1)[0], BATCH_SIZE), "real_lables"); realLabels.Zero();
        one.FuseSubTensor2D(1, 0, realLabels);

        // setup data preloader
        Pix2PixImageLoader loader(trainFiles, BATCH_SIZE, 1, 1337);
        DataPreloader preloader({ &condImages, &realImages }, { &loader }, 5, false);

        for (uint32_t e = 1; e <= EPOCHS; ++e)
        {
            preloader.Load();

            // generate fake images from condition
            //Tensor fakeImages = *gModel->Predict(condImages)[0];
            Tensor fakeImages(Shape::From(IMG_SHAPE, BATCH_SIZE));
            fakeImages.FillWithFunc([]() { return Uniform::NextSingle(-1, 1); });

            auto realTrainData = dModel->TrainOnBatch({ &realImages }, { &realLabels });
            //cout << get<0>(realTrainData) << endl;
            auto fakeTrainData = dModel->TrainOnBatch({ &fakeImages }, { &fakeLabels });
            //cout << get<0>(fakeTrainData) << endl;

            cout << ">" << e << setprecision(4) << fixed << " loss=" << (get<0>(realTrainData) + get<0>(fakeTrainData)) * 0.5f << " real=" << round(get<1>(realTrainData) * 100) << "% fake=" << round(get<1>(fakeTrainData) * 100) << "%" << endl;
        }

        cin.get();
    }

    ModelBase* CreateGenerator(const Shape& imgShape);
    ModelBase* CreatePatchDiscriminator(const Shape& imgShape, uint32_t patchSize, bool useMiniBatchDiscrimination = true);
};
