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
        GlobalRngSeed(1337);

        Shape IMG_SHAPE = Shape(256, 256, 3);

        cout << "Example: Pix2Pix" << endl;

        auto trainFiles = LoadFilesList("f:/!TrainingData/maps/train", false, true);

        Tensor inImages(Shape::From(IMG_SHAPE, (uint32_t)trainFiles.size()));
        Tensor outImages(Shape::From(IMG_SHAPE, (uint32_t)trainFiles.size()));

        // pre-process training data
        {
            Tensor t1(IMG_SHAPE), t2(IMG_SHAPE);
            tensor_ptr_vec_t output{ &t1, &t2 };
            for (size_t i = 0; i < trainFiles.size(); ++i)
            {
                auto& filename = trainFiles[i];
                auto img = LoadImage(filename, IMG_SHAPE.Width() * 2 * 1, IMG_SHAPE.Height() * 1, IMG_SHAPE.Width() * 2, IMG_SHAPE.Height());

                img.Split(WidthAxis, output);

                t1.CopyBatchTo(0, i, inImages);
                t2.CopyBatchTo(0, i, outImages);
                /*t1.SaveAsImage("1.jpg", false);
                t2.SaveAsImage("2.jpg", false);*/
            }
        }

        inImages.SaveAsImage("inImages.png", false);
        outImages.SaveAsImage("outImages.png", false);

        auto gModel = CreateGenerator(IMG_SHAPE);
        //cout << "Generator" << endl << gModel->Summary();
        auto dModel = CreateDiscriminator(IMG_SHAPE);
        //cout << "Discriminator" << endl << dModel->Summary();

        auto inSrc = new Input(IMG_SHAPE);
        auto genOut = gModel->Call(inSrc->Outputs());
        auto disOut = dModel->Call({ inSrc->Outputs()[0], genOut[0] });

        auto ganModel = new Flow(inSrc->Outputs(), { disOut[0], genOut[0] }, "pix2pix");
        ganModel->Optimize(new Adam(0.0002f, 0.5f), { new BinaryCrossEntropy(), new MeanAbsoluteError() }, { 1.f, 100.f });
        //cout << "GAN" << endl << ganModel->Summary();

        //Tensor trainingData[2];

        //LoadImages(images);
        //m_ImageShape = Shape(images.Width(), images.Height(), images.Depth());
        //images.Map([](float x) { return (x - 127.5f) / 127.5f; }, images);
        //images.Reshape(Shape::From(dModel->InputShapesAt(-1)[0], images.Batch()));
        //images.Name("images");

        //const uint32_t BATCH_SIZE = 1;
        //const uint32_t STEPS = 100000;



        //Tensor testNoise(Shape::From(gModel->InputShapesAt(-1)[0], 100), "test_noise"); testNoise.FillWithFunc([]() { return Normal::NextSingle(0, 1); });

        //Tensor noise(Shape::From(gModel->InputShapesAt(-1)[0], BATCH_SIZE), "noise");
        //Tensor real(Shape::From(dModel->OutputShapesAt(-1)[0], BATCH_SIZE), "real"); real.FillWithValue(1.f);

        //Tensor noiseHalf(Shape::From(gModel->InputShapesAt(-1)[0], BATCH_SIZE / 2), "noise_half");
        //Tensor realHalf(Shape::From(dModel->OutputShapesAt(-1)[0], BATCH_SIZE / 2), "real_half"); realHalf.FillWithValue(1.f);
        //Tensor fakeHalf(Shape::From(dModel->OutputShapesAt(-1)[0], BATCH_SIZE / 2), "fake_half"); fakeHalf.FillWithValue(0.f);

        //float totalGanLoss = 0.f;

        //Tqdm progress(STEPS, 0);
        //progress.ShowEta(true).ShowElapsed(false).ShowPercent(false);
        //for (uint32_t i = 0; i < STEPS; ++i, progress.NextStep())
        //{
        //    noiseHalf.FillWithFunc([]() { return Normal::NextSingle(0, 1); });

        //    // generate fake images from noise
        //    Tensor fakeImages = *gModel->Predict(noiseHalf)[0];
        //    fakeImages.Name("fake_images");
        //    // grab random batch of real images
        //    Tensor realImages = images.GetRandomBatches(BATCH_SIZE / 2);
        //    realImages.Name("real_images");

        //    // perform step of training discriminator to distinguish fake from real images
        //    dModel->SetTrainable(true);
        //    float dRealLoss = get<0>(dModel->TrainOnBatch(realImages, realHalf));
        //    float dFakeLoss = get<0>(dModel->TrainOnBatch(fakeImages, fakeHalf));

        //    noise.FillWithFunc([]() { return Normal::NextSingle(0, 1); });

        //    // perform step of training generator to generate more real images (the more discriminator is confident that a particular image is fake the more generator will learn)
        //    dModel->SetTrainable(false);
        //    float ganLoss = get<0>(ganModel->TrainOnBatch(noise, real));

        //    stringstream extString;
        //    extString << setprecision(4) << fixed << " - real_l: " << dRealLoss << " - fake_l: " << dFakeLoss << " - gan_l: " << ganLoss;
        //    progress.SetExtraString(extString.str());

        //    if (i % 50 == 0)
        //        gModel->Predict(testNoise)[0]->Map([](float x) { return x * 127.5f + 127.5f; }).Reshaped(Shape(m_ImageShape.Width(), m_ImageShape.Height(), m_ImageShape.Depth(), -1)).SaveAsImage(Name() + "_e" + PadLeft(to_string(e), 4, '0') + "_b" + PadLeft(to_string(i), 4, '0') + ".png", false);
        //}

        ganModel->SaveWeights("pix2pix.h5");
    }

    ModelBase* CreateGenerator(const Shape& imgShape)
    {
        auto encoderBlock = [](TensorLike* input, uint32_t filtersNum, bool batchNorm = true)
        {
            InitializerBase* init = new Normal(0.f, 0.02f);

            auto g = (new Conv2D(filtersNum, 4, 2, Tensor::GetPadding(Same, 4)))->KernelInitializer(init)->Call(input);
            if (batchNorm)
                g = (new BatchNormalization())->Call(g);
            g = (new Activation(new LeakyReLU(0.2f)))->Call(g);
            return g;
        };

        auto decoderBlock = [](TensorLike* input, TensorLike* skipInput, uint32_t filtersNum, bool dropout = true)
        {
            InitializerBase* init = new Normal(0.f, 0.02f);

            auto g = (new Conv2DTranspose(filtersNum, 4, 2, Tensor::GetPadding(Same, 4)))->KernelInitializer(init)->Call(input);
            g = (new BatchNormalization())->Call(g);
            if (dropout)
                g = (new Dropout(0.5f))->Call(g);
            g = (new Concatenate(DepthAxis))->Call({ g[0], skipInput });
            g = (new Activation(new ReLU()))->Call(g);
            return g;
        };

        InitializerBase* init = new Normal(0.f, 0.02f);

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
        auto b = (new Conv2D(512, 4, 2, Tensor::GetPadding(Same, 4)))->KernelInitializer(init)->Call(e7);
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
        auto g = (new Conv2DTranspose(3, 4, 2, Tensor::GetPadding(Same, 4)))->KernelInitializer(init)->Call(d7);
        auto outImage = (new Activation(new Tanh()))->Call(g);

        auto model = new Flow({ inImage->Outputs()[0] }, outImage);
        return model;
    }
    
    ModelBase* CreateDiscriminator(const Shape& imgShape)
    {
        InitializerBase* init = new Normal(0.f, 0.02f);

        auto inSrcImage = new Input(imgShape);
        auto inTargetImage = new Input(imgShape);

        auto merged = (new Concatenate(DepthAxis))->Call({ inSrcImage->Outputs()[0], inTargetImage->Outputs()[0] });
        // 64
        auto d = (new Conv2D(64, 4, 2, Tensor::GetPadding(Same, 4), new LeakyReLU(0.2f)))->KernelInitializer(init)->Call(merged);
        // 128
        d = (new Conv2D(128, 4, 2, Tensor::GetPadding(Same, 4)))->KernelInitializer(init)->Call(d);
        d = (new BatchNormalization())->Call(d);
        d = (new Activation(new LeakyReLU(0.2f)))->Call(d);
        // 256
        d = (new Conv2D(256, 4, 2, Tensor::GetPadding(Same, 4)))->KernelInitializer(init)->Call(d);
        d = (new BatchNormalization())->Call(d);
        d = (new Activation(new LeakyReLU(0.2f)))->Call(d);
        // 512
        d = (new Conv2D(512, 4, 2, Tensor::GetPadding(Same, 4)))->KernelInitializer(init)->Call(d);
        d = (new BatchNormalization())->Call(d);
        d = (new Activation(new LeakyReLU(0.2f)))->Call(d);
        d = (new Conv2D(512, 4, 1, Tensor::GetPadding(Same, 4)))->KernelInitializer(init)->Call(d);
        d = (new BatchNormalization())->Call(d);
        d = (new Activation(new LeakyReLU(0.2f)))->Call(d);
        // patch output
        d = (new Conv2D(1, 4, 1, Tensor::GetPadding(Same, 4)))->KernelInitializer(init)->Call(d);
        auto patchOut = (new Activation(new Sigmoid()))->Call(d);

        auto model = new Flow({ inSrcImage->Outputs()[0], inTargetImage->Outputs()[0] }, patchOut);
        auto opt = new Adam(0.0002f, 0.5f);
        model->Optimize(opt, new BinaryCrossEntropy(), { 0.5f });
        return model;
    }
};
