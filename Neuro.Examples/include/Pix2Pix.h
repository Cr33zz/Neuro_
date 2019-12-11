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
    struct EdgeImageLoader : public ImageLoader
    {
        EdgeImageLoader(const vector<string>& files, uint32_t batchSize, uint32_t upScaleFactor = 1, uint32_t seed = 0) :ImageLoader(files, batchSize, upScaleFactor), m_Rng(seed) {}

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

    // Splits source image into 2 images along width axis
    struct SplitImageLoader : public ImageLoader
    {
        SplitImageLoader(const vector<string>& files, uint32_t batchSize, uint32_t seed = 0) :ImageLoader(files, batchSize), m_Rng(seed) {}

        virtual size_t operator()(vector<Tensor>& dest, size_t loadIdx) override
        {
            auto& img1 = dest[loadIdx];
            auto& img2 = dest[loadIdx + 1];

            NEURO_ASSERT(img1.Width() == img2.Width(), "");
            NEURO_ASSERT(img1.Height() == img2.Height(), "");

            img1.ResizeBatch(m_BatchSize);
            img1.OverrideHost();
            img2.ResizeBatch(m_BatchSize);
            img2.OverrideHost();

            Tensor t1(Shape::From(img1.GetShape(), 1));
            Tensor t2(Shape::From(img2.GetShape(), 1));
            tensor_ptr_vec_t tmp{ &t1, &t2 };

            for (uint32_t n = 0; n < m_BatchSize; ++n)
            {
                const auto& file = m_Files[m_Rng.Next((int)m_Files.size())];
                auto img = LoadImage(file, img1.Width() * 2, img1.Height());
                img.Split(WidthAxis, tmp);

                t1.Sub(127.5f).Div(127.5f).CopyBatchTo(0, (uint32_t)n, img1);
                t2.Sub(127.5f).Div(127.5f).CopyBatchTo(0, (uint32_t)n, img2);
            }

            img1.CopyToDevice();
            img2.CopyToDevice();
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
        const int EPOCHS = 150;
        const uint32_t BATCH_SIZE = 1;
        const float LEARNING_RATE = 0.0002f;
        const float ADAM_BETA1 = 0.5f;
        //const uint32_t STEPS = 6;

        cout << "Example: Pix2Pix" << endl;

        //const string NAME = "flowers";
        //auto trainFiles = LoadFilesList("data/flowers", false, true);
        const string NAME = "facades";
        auto trainFiles = LoadFilesList("data/facades/train", false, true);

        // setup models
        auto gModel = CreateGenerator(IMG_SHAPE);
        cout << "Generator" << endl << gModel->Summary();
        auto dModel = CreateDiscriminator(IMG_SHAPE);
        //dModel->Optimize(new Adam(LEARNING_RATE, ADAM_BETA1), new BinaryCrossEntropy(), {}, All);
        //cout << "Discriminator" << endl << dModel->Summary();

        //auto inSrc = (new Input(IMG_SHAPE))->Outputs()[0];
        auto inputImg = new Placeholder(IMG_SHAPE);
        auto targetImg = new Placeholder(IMG_SHAPE);
        
        auto genImg = gModel->Call(inputImg, "generator")[0];
        auto disOut1 = dModel->Call({ inputImg, genImg }, "discriminator1")[0];
        auto disOut2 = dModel->Call({ inputImg, targetImg }, "discriminator2")[0];

        auto fakeLabels = new Constant(zeros(disOut1->GetShape()), "fake_labels");
        auto realLabels = new Constant(ones(disOut2->GetShape()), "real_labels");
        auto crossEntropy = new BinaryCrossEntropy();

        auto discLoss = add(crossEntropy->Build(fakeLabels, disOut1), crossEntropy->Build(realLabels, disOut2));

        auto ganLoss = crossEntropy->Build(realLabels, disOut1);
        auto l1Loss = mean(abs(sub(targetImg, genImg)));
        auto genLoss = add(ganLoss, multiply(l1Loss, 100.f));

        auto genOpt = new Adam(LEARNING_RATE, ADAM_BETA1);
        auto genMinimize = genOpt->Minimize({ genLoss });
        auto discOpt = new Adam(LEARNING_RATE, ADAM_BETA1);
        auto discMinimize = discOpt->Minimize({ discLoss });

        dModel->LoadWeights(NAME + "_disc.h5", false, true);
        gModel->LoadWeights(NAME + "_gen.h5", false, true);

        // setup data preloader
        //EdgeImageLoader loader(trainFiles, BATCH_SIZE, 1);
        //DataPreloader preloader({ &condImages, &realImages }, { &loader }, 5);
        SplitImageLoader loader(trainFiles, BATCH_SIZE, 1); // facades
        DataPreloader preloader({ targetImg->OutputPtr(), inputImg->OutputPtr() }, { &loader }, 5); // facades

        const size_t STEPS = trainFiles.size();

        for (size_t e = 0; e < EPOCHS; ++e)
        {
            Tqdm progress(STEPS, 0);
            progress.ShowEta(true).ShowElapsed(false).ShowPercent(false);
            for (size_t s = 0; s < STEPS; ++s, progress.NextStep())
            {
                //load next conditional and expected images
                preloader.Load();

                // optimize discriminator
                dModel->SetTrainable(true);
                gModel->SetTrainable(false);
                auto discResults = Session::Default()->Run({ discLoss, discMinimize }, {});
                float dLoss = (*discResults[0])(0);

                // optimize generator
                dModel->SetTrainable(false);
                gModel->SetTrainable(true);
                auto genResults = Session::Default()->Run({ genLoss, ganLoss, l1Loss, genImg, genMinimize }, {});
                float _genLoss = (*genResults[0])(0);
                float _ganLoss = (*genResults[1])(0);
                float _l1Loss = (*genResults[2])(0);
                auto _genImg = (*genResults[3]);

                if ((e * STEPS + s) % 50 == 0)
                {
                    dModel->SaveWeights(NAME + "_disc.h5");
                    gModel->SaveWeights(NAME + "_gen.h5");
                    Tensor tmp(Shape(IMG_SHAPE.Width() * 3, IMG_SHAPE.Height(), IMG_SHAPE.Depth(), BATCH_SIZE));
                    Tensor::Concat(WidthAxis, { inputImg->OutputPtr(), &_genImg, targetImg->OutputPtr() }, tmp);
                    tmp.Add(1.f).Mul(127.5f).SaveAsImage(NAME + "_s" + PadLeft(to_string(e * STEPS + s), 4, '0') + ".jpg", false, 1);
                }

                stringstream extString;
                extString << setprecision(4) << fixed << " - dLoss: " << dLoss << " - ganLoss: " << _ganLoss << " - l1Loss: " << _l1Loss << " - genLoss: " << _genLoss;
                progress.SetExtraString(extString.str());
            }
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
