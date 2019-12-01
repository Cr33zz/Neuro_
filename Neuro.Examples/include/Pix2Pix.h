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
    void Run()
    {
        Tensor::SetDefaultOpMode(GPU);
        //GlobalRngSeed(1337);

        Shape IMG_SHAPE = Shape(256, 256, 3);

        cout << "Example: Pix2Pix" << endl;

        auto trainFiles = LoadFilesList("f:/!TrainingData/flowers", false, true);

        Tensor inImages(Shape::From(IMG_SHAPE, (uint32_t)trainFiles.size()));
        Tensor outImages(Shape::From(IMG_SHAPE, (uint32_t)trainFiles.size()));

        // pre-process training data
        //{
        //    Tensor t1(IMG_SHAPE), t2(IMG_SHAPE);
        //    tensor_ptr_vec_t output{ &t1, &t2 };

        //    cout << "Pre-processing training data" << endl;
        //    Tqdm progress(trainFiles.size(), 0);
        //    for (size_t i = 0; i < trainFiles.size(); ++i, progress.NextStep())
        //    {
        //        auto& filename = trainFiles[i];

        //        auto img = LoadImage(filename, IMG_SHAPE.Width(), IMG_SHAPE.Height());
        //        t1 = CannyEdgeDetection(img).ToRGB();
        //        t2 = img;
        //        
        //        //auto img = LoadImage(filename, IMG_SHAPE.Width() * 2 * 1, IMG_SHAPE.Height() * 1, IMG_SHAPE.Width() * 2, IMG_SHAPE.Height());
        //        //actual pre-process
        //        //img.Split(WidthAxis, output);
        //        t1.Sub(127.5f).Div(127.5f).CopyBatchTo(0, (uint32_t)i, inImages);
        //        t2.Sub(127.5f).Div(127.5f).CopyBatchTo(0, (uint32_t)i, outImages);
        //    }

        //    /*inImages.SaveAsImage("__in.jpg", false);
        //    outImages.SaveAsImage("__out.jpg", false);*/

        //    inImages.SaveAsH5("inImages.h5");
        //    outImages.SaveAsH5("outImages.h5");
        //}
        // load pre-processed data
        {
            inImages.LoadFromH5("inImages.h5");
            outImages.LoadFromH5("outImages.h5");
        }

        //inImages.Add(1.f).Mul(127.5f).SaveAsImage("_in.jpg", false);

        auto gModel = CreateGenerator(IMG_SHAPE);
        //cout << "Generator" << endl << gModel->Summary();
        auto dModel = CreateDiscriminator(IMG_SHAPE);
        //cout << "Discriminator" << endl << dModel->Summary();

        auto inSrc = new Input(IMG_SHAPE);
        auto genOut = gModel->Call(inSrc->Outputs());
        auto disOut = dModel->Call({ inSrc->Outputs()[0], genOut[0] });

        auto ganModel = new Flow(inSrc->Outputs(), { disOut[0], genOut[0] }, "pix2pix");
        ganModel->Optimize(new Adam(0.0002f, 0.5f), { new BinaryCrossEntropy(), new MeanAbsoluteError() }, { 1.f, 100.f });
        ganModel->LoadWeights("pix2pix.h5", false, true);

        const uint32_t BATCH_SIZE = 1;
        const uint32_t STEPS = 100000;
        //const uint32_t STEPS = 6;

        Tensor condImages(Shape::From(gModel->InputShapesAt(-1)[0], BATCH_SIZE), "cond_image");
        Tensor expectedImages(Shape::From(gModel->OutputShapesAt(-1)[0], BATCH_SIZE), "output_image");

        Tensor real(Shape::From(dModel->OutputShapesAt(-1)[0], BATCH_SIZE), "real"); real.One();
        Tensor fake(Shape::From(dModel->OutputShapesAt(-1)[0], BATCH_SIZE), "fake"); fake.Zero();

        Tqdm progress(STEPS, 0);
        progress.ShowEta(true).ShowElapsed(false).ShowPercent(false);
        for (uint32_t i = 0; i < STEPS; ++i, progress.NextStep())
        {
            //select random batch indices
            vector<uint32_t> batchesIdx(BATCH_SIZE);
            generate(batchesIdx.begin(), batchesIdx.end(), [&]() { return (uint32_t)GlobalRng().Next(inImages.Batch()); });

            inImages.GetBatches(batchesIdx, condImages);
            outImages.GetBatches(batchesIdx, expectedImages);

            // generate fake images from condition
            Tensor fakeImages = *gModel->Predict(condImages)[0];

            //Debug::LogAllOutputs(true);
            //Debug::LogAllGrads(true);

            // perform step of training discriminator to distinguish fake from real images
            dModel->SetTrainable(true);
            float dRealLoss = get<0>(dModel->TrainOnBatch({ &condImages, &expectedImages }, { &real }));
            float dFakeLoss = get<0>(dModel->TrainOnBatch({ &condImages, &fakeImages }, { &fake }));

            // perform step of training generator to generate more real images (the more discriminator is confident that a particular image is fake the more generator will learn)
            dModel->SetTrainable(false);
            float ganLoss = get<0>(ganModel->TrainOnBatch({ &condImages }, { &real, &expectedImages }));

            //Debug::LogAllOutputs(false);
            //Debug::LogAllGrads(false);

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
            auto g = (new ZeroPadding2D(2, 1, 2, 1))->Call(input);
            g = (new Conv2D(filtersNum, 4, 2))->Call(g);
            if (batchNorm)
                g = (new BatchNormalization())->Call(g);
            g = (new Activation(new LeakyReLU(0.2f)))->Call(g);
            return g;
        };

        auto decoderBlock = [](TensorLike* input, TensorLike* skipInput, uint32_t filtersNum, bool dropout = true)
        {
            auto g = (new UpSampling2D(2))->Call(input);
            g = (new ZeroPadding2D(2, 1, 2, 1))->Call(g);
            g = (new Conv2D(filtersNum, 4))->Call(g);
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
        auto b = (new ZeroPadding2D(2, 1, 2, 1))->Call(e7);
        b = (new Conv2D(512, 4, 2))->Call(b);
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
        g = (new ZeroPadding2D(2, 1, 2, 1))->Call(g);
        g = (new Conv2D(3, 4))->Call(g);
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
        auto d = (new ZeroPadding2D(2, 1, 2, 1))->Call(merged);
        d = (new Conv2D(64, 4, 2, 0, new LeakyReLU(0.2f)))->Call(d);
        // 128
        d = (new ZeroPadding2D(2, 1, 2, 1))->Call(d);
        d = (new Conv2D(128, 4, 2))->Call(d);
        d = (new BatchNormalization())->Call(d);
        d = (new Activation(new LeakyReLU(0.2f)))->Call(d);
        // 256
        d = (new ZeroPadding2D(2, 1, 2, 1))->Call(d);
        d = (new Conv2D(256, 4, 2))->Call(d);
        d = (new BatchNormalization())->Call(d);
        d = (new Activation(new LeakyReLU(0.2f)))->Call(d);
        // 512
        d = (new ZeroPadding2D(2, 1, 2, 1))->Call(d);
        d = (new Conv2D(512, 4, 2))->Call(d);
        d = (new BatchNormalization())->Call(d);
        d = (new Activation(new LeakyReLU(0.2f)))->Call(d);
        d = (new ZeroPadding2D(2, 1, 2, 1))->Call(d);
        d = (new Conv2D(512, 4, 1))->Call(d);
        d = (new BatchNormalization())->Call(d);
        d = (new Activation(new LeakyReLU(0.2f)))->Call(d);
        // patch output
        d = (new ZeroPadding2D(2, 1, 2, 1))->Call(d);
        d = (new Conv2D(1, 4, 1))->Call(d);
        auto patchOut = (new Activation(new Sigmoid()))->Call(d);

        auto model = new Flow({ inSrcImage->Outputs()[0], inTargetImage->Outputs()[0] }, patchOut);
        auto opt = new Adam(0.0002f, 0.5f);
        model->Optimize(opt, new BinaryCrossEntropy(), { 0.5f });
        return model;
    }
};
