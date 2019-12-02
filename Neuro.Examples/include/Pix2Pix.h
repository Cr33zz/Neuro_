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
        Pix2PixImageLoader(const vector<string>& files, uint32_t batchSize, uint32_t upScaleFactor = 1) :ImageLoader(files, batchSize, upScaleFactor) {}

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
                auto img = LoadImage(m_Files[GlobalRng().Next((int)m_Files.size())], cndImg.Width() * m_UpScaleFactor, cndImg.Height() * m_UpScaleFactor, cndImg.Width(), cndImg.Height());
                auto edges = CannyEdgeDetection(img).ToRGB();
                
                edges.Sub(127.5f).Div(127.5f).CopyBatchTo(0, (uint32_t)n, cndImg);
                img.Sub(127.5f).Div(127.5f).CopyBatchTo(0, (uint32_t)n, outImg);                
            }

            cndImg.CopyToDevice();
            outImg.CopyToDevice();
            return 2;
        }
    };

    void Run()
    {
        Tensor::SetDefaultOpMode(GPU);
        //GlobalRngSeed(1337);

        const Shape IMG_SHAPE = Shape(256, 256, 3);
        const uint32_t BATCH_SIZE = 1;
        const uint32_t STEPS = 100000;
        //const uint32_t STEPS = 6;

        cout << "Example: Pix2Pix" << endl;

        auto trainFiles = LoadFilesList("f:/!TrainingData/flowers", false, true);

        Tensor condImages(Shape::From(IMG_SHAPE, BATCH_SIZE), "cond_image");
        Tensor expectedImages(Shape::From(IMG_SHAPE, BATCH_SIZE), "output_image");

        // setup data preloader
        Pix2PixImageLoader loader(trainFiles, BATCH_SIZE, 2);
        DataPreloader preloader({ &condImages, &expectedImages }, { &loader }, 5);

        // setup models
        auto gModel = CreateGenerator(IMG_SHAPE);
        cout << "Generator" << endl << gModel->Summary();
        auto dModel = CreateDiscriminator(IMG_SHAPE);
        cout << "Discriminator" << endl << dModel->Summary();

        auto inSrc = new Input(IMG_SHAPE);
        auto genOut = gModel->Call(inSrc->Outputs());
        auto disOut = dModel->Call({ inSrc->Outputs()[0], genOut[0] });

        auto ganModel = new Flow(inSrc->Outputs(), { disOut[0], genOut[0] }, "pix2pix");
        ganModel->Optimize(new Adam(0.0002f, 0.5f), { new BinaryCrossEntropy(), new MeanAbsoluteError() }, { 1.f, 100.f });
        ganModel->LoadWeights("pix2pix.h5", false, true);

        Tensor real(Shape::From(dModel->OutputShapesAt(-1)[0], BATCH_SIZE), "real"); real.One();
        Tensor fake(Shape::From(dModel->OutputShapesAt(-1)[0], BATCH_SIZE), "fake"); fake.Zero();

        Tqdm progress(STEPS, 0);
        progress.ShowEta(true).ShowElapsed(false).ShowPercent(false);
        for (uint32_t i = 0; i < STEPS; ++i, progress.NextStep())
        {
            //load next conditional and expected images
            preloader.Load();

            // generate fake images from condition
            Tensor fakeImages = *gModel->Predict(condImages)[0];

            // perform step of training discriminator to distinguish fake from real images
            dModel->SetTrainable(true);
            float dRealLoss = get<0>(dModel->TrainOnBatch({ &condImages, &expectedImages }, { &real }));
            float dFakeLoss = get<0>(dModel->TrainOnBatch({ &condImages, &fakeImages }, { &fake }));

            // perform step of training generator to generate more real images
            dModel->SetTrainable(false);
            float ganLoss = get<0>(ganModel->TrainOnBatch({ &condImages }, { &real, &expectedImages }));

            stringstream extString;
            extString << setprecision(4) << fixed << " - real_l: " << dRealLoss << " - fake_l: " << dFakeLoss << " - gan_l: " << ganLoss;
            progress.SetExtraString(extString.str());

            if (i % 100 == 0)
            {
                ganModel->SaveWeights("pix2pix.h5");
                Tensor tmp(Shape::From(fakeImages.GetShape(), 3));
                Tensor::Concat(BatchAxis, { &condImages, &fakeImages, &expectedImages }, tmp);
                tmp.Add(1.f).Mul(127.5f).SaveAsImage("pix2pix_s" + PadLeft(to_string(i), 4, '0') + ".jpg", false);
            }
        }

        ganModel->SaveWeights("pix2pix.h5");
    }

    ModelBase* CreateGenerator(const Shape& imgShape)
    {
        auto encoderBlock = [](TensorLike* input, uint32_t filtersNum, bool batchNorm = true)
        {
            //auto g = (new ZeroPadding2D(1, 1, 1, 1))->Call(input);
            auto g = (new Conv2D(filtersNum, 3, 2, Tensor::GetPadding(Same, 3)))->Call(input);
            if (batchNorm)
                g = (new BatchNormalization())->Call(g);
            g = (new Activation(new LeakyReLU(0.2f)))->Call(g);
            return g;
        };

        auto decoderBlock = [](TensorLike* input, TensorLike* skipInput, uint32_t filtersNum, bool dropout = true)
        {
            auto g = (new UpSampling2D(2))->Call(input);
            //g = (new ZeroPadding2D(2, 2, 2, 2))->Call(g);
            g = (new Conv2D(filtersNum, 3, 1, Tensor::GetPadding(Same, 3)))->Call(g);
            g = (new BatchNormalization())->Call(g);
            if (dropout)
                g = (new Dropout(0.5f))->Call(g);
            g = (new Concatenate(DepthAxis))->Call({ g[0], skipInput });
            g = (new Activation(new ReLU()))->Call(g);
            return g;
        };

        auto inImage = new Input(imgShape);

        ///encoder
        auto e1 = encoderBlock(inImage->Outputs()[0], 64, false);
        auto e2 = encoderBlock(e1[0], 128);
        auto e3 = encoderBlock(e2[0], 256);
        auto e4 = encoderBlock(e3[0], 512);
        auto e5 = encoderBlock(e4[0], 512);
        auto e6 = encoderBlock(e5[0], 512);
        auto e7 = encoderBlock(e6[0], 512);
        /// bottleneck
        //auto b = (new ZeroPadding2D(2, 2, 2, 2))->Call(e7);
        auto b = (new Conv2D(512, 3, 2, Tensor::GetPadding(Same, 3)))->Call(e7);
        b = (new Activation(new ReLU()))->Call(b);
        /// decoder
        auto d1 = decoderBlock(b[0], e7[0], 512);
        auto d2 = decoderBlock(d1[0], e6[0], 512);
        auto d3 = decoderBlock(d2[0], e5[0], 512);
        auto d4 = decoderBlock(d3[0], e4[0], 512, false);
        auto d5 = decoderBlock(d4[0], e3[0], 256, false);
        auto d6 = decoderBlock(d5[0], e2[0], 128, false);
        auto d7 = decoderBlock(d6[0], e1[0], 64, false);
        /// output
        auto g = (new UpSampling2D(2))->Call(d7);
        //g = (new ZeroPadding2D(2, 2, 2, 2))->Call(g);
        g = (new Conv2D(3, 3, 1, Tensor::GetPadding(Same, 3)))->Call(g);
        auto outImage = (new Activation(new Tanh()))->Call(g);

        auto model = new Flow({ inImage->Outputs()[0] }, outImage);
        return model;
    }
    
    ModelBase* CreateDiscriminator(const Shape& imgShape)
    {
        auto inSrcImage = new Input(imgShape);
        auto inTargetImage = new Input(imgShape);

        auto merged = (new Concatenate(DepthAxis))->Call({ inSrcImage->Outputs()[0], inTargetImage->Outputs()[0] });
        // 64
        //auto d = (new ZeroPadding2D(2, 2, 2, 2))->Call(merged);
        auto d = (new Conv2D(64, 3, 2, Tensor::GetPadding(Same, 3), new LeakyReLU(0.2f)))->Call(merged);
        // 128
        //d = (new ZeroPadding2D(2, 2, 2, 2))->Call(d);
        d = (new Conv2D(128, 3, 2, Tensor::GetPadding(Same, 3)))->Call(d);
        d = (new BatchNormalization())->Call(d);
        d = (new Activation(new LeakyReLU(0.2f)))->Call(d);
        // 256
        //d = (new ZeroPadding2D(2, 2, 2, 2))->Call(d);
        d = (new Conv2D(256, 3, 2, Tensor::GetPadding(Same, 3)))->Call(d);
        d = (new BatchNormalization())->Call(d);
        d = (new Activation(new LeakyReLU(0.2f)))->Call(d);
        // 512
        //d = (new ZeroPadding2D(2, 2, 2, 2))->Call(d);
        d = (new Conv2D(512, 3, 2, Tensor::GetPadding(Same, 3)))->Call(d);
        d = (new BatchNormalization())->Call(d);
        d = (new Activation(new LeakyReLU(0.2f)))->Call(d);
        //d = (new ZeroPadding2D(2, 2, 2, 2))->Call(d);
        d = (new Conv2D(512, 3, 1, Tensor::GetPadding(Same, 3)))->Call(d);
        d = (new BatchNormalization())->Call(d);
        d = (new Activation(new LeakyReLU(0.2f)))->Call(d);
        // patch output
        //d = (new ZeroPadding2D(2, 2, 2, 2))->Call(d);
        d = (new Conv2D(1, 3, 1, Tensor::GetPadding(Same, 3)))->Call(d);
        auto patchOut = (new Activation(new Sigmoid()))->Call(d);

        auto model = new Flow({ inSrcImage->Outputs()[0], inTargetImage->Outputs()[0] }, patchOut);
        auto opt = new Adam(0.0002f, 0.5f);
        model->Optimize(opt, new BinaryCrossEntropy(), { 0.5f });
        return model;
    }
};
