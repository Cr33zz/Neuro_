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
        Tensor expectedImages(Shape::From(IMG_SHAPE, BATCH_SIZE), "output_image");

        // setup data preloader
        Pix2PixImageLoader loader(trainFiles, BATCH_SIZE, 1);
        DataPreloader preloader({ &condImages, &expectedImages }, { &loader }, 5);

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
                dLoss = get<0>(dModel->TrainOnBatch({ &condImages, &expectedImages }, { &realLabels }));
            }
            else
            {
                // generate fake images from condition
                Tensor fakeImages = *gModel->Predict(condImages)[0];
                dLoss = get<0>(dModel->TrainOnBatch({ &condImages, &fakeImages }, { &fakeLabels }));

                if (discriminatorTrainingSource % 10 == 0)
                {
                    ganModel->SaveWeights("pix2pix.h5");
                    Tensor tmp(Shape(IMG_SHAPE.Width() * 3, IMG_SHAPE.Height(), IMG_SHAPE.Depth(), BATCH_SIZE));
                    Tensor::Concat(WidthAxis, { &condImages, &fakeImages, &expectedImages }, tmp);
                    tmp.Add(1.f).Mul(127.5f).SaveAsImage("pix2pix_s" + PadLeft(to_string(i), 4, '0') + ".jpg", false);
                }
            }
            ++discriminatorTrainingSource;
            
            // perform step of training generator to generate more real images
            dModel->SetTrainable(false);
            float ganLoss = get<0>(ganModel->TrainOnBatch({ &condImages }, { &realLabels, &expectedImages }));

            stringstream extString;
            extString << setprecision(4) << fixed << " - disc_l: " << dLoss << " - gan_l: " << ganLoss;
            progress.SetExtraString(extString.str());
        }

        ganModel->SaveWeights("pix2pix.h5");
    }

    void RunDiscriminatorTrainTest()
    {
        Tensor::SetDefaultOpMode(GPU);

        //GlobalRngSeed(1337);

        const Shape IMG_SHAPE(256, 256, 3);
        const uint32_t PATCH_SIZE = 64;
        const uint32_t BATCH_SIZE = 1;
        const uint32_t EPOCHS = 30;

        auto trainFiles = LoadFilesList("e:/Downloads/flowers", false, true);

        Tensor condImages(Shape::From(IMG_SHAPE, BATCH_SIZE), "cond_image");
        Tensor expectedImages(Shape::From(IMG_SHAPE, BATCH_SIZE), "output_image");

        // setup data preloader
        Pix2PixImageLoader loader(trainFiles, BATCH_SIZE, 1);
        DataPreloader preloader({ &condImages, &expectedImages }, { &loader }, 5);

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

        for (uint32_t e = 1; e <= EPOCHS; ++e)
        {
            preloader.Load();

            // generate fake images from condition
            Tensor fakeImages = *gModel->Predict(condImages)[0];
            /*Tensor fakeImages(Shape::From(dModel->InputShapesAt(-1)[0], BATCH_SIZE)); fakeImages.FillWithFunc([]() { return Uniform::NextSingle(-1, 1); });
            Tensor realImages = images.GetRandomBatches(BATCH_SIZE);*/

            auto realTrainData = dModel->TrainOnBatch({ &condImages, &expectedImages }, { &realLabels });
            auto fakeTrainData = dModel->TrainOnBatch({ &condImages, &fakeImages }, { &fakeLabels });

            cout << ">" << e << setprecision(4) << fixed << " loss=" << (get<0>(realTrainData) + get<0>(fakeTrainData)) * 0.5f << " real=" << round(get<1>(realTrainData) * 100) << "% fake=" << round(get<1>(fakeTrainData) * 100) << "%" << endl;
        }

        cin.get();
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
    
    ModelBase* CreatePatchDiscriminator(const Shape& imgShape, uint32_t patchSize, bool useMiniBatchDiscrimination = true)
    {
        NEURO_ASSERT(imgShape.Width() == imgShape.Height(), "Input image must be square.");
        NEURO_ASSERT(imgShape.Width() % patchSize == 0, "Input image size is not divisible by patch size.");

        size_t nbPatches = (size_t)::pow(imgShape.Width() / patchSize, 2);
        uint32_t stride = 2;
        auto patchInput = new Input(Shape(patchSize, patchSize, imgShape.Depth()));

        uint32_t filtersStart = 64;
        size_t nbConv = int(floor(::log(patchSize) / ::log(2)));
        vector<uint32_t> filtersList(nbConv);
        for (int i = 0; i < nbConv; ++i)
            filtersList[i] = filtersStart * (uint32_t)min(8, ::pow(2, i));

        auto discOut = (new Conv2D(filtersList[0], 3, stride, Tensor::GetPadding(Same, 3), new LeakyReLU(0.2f)))->Call(patchInput->Outputs());

        for (uint32_t i = 1; i < filtersList.size(); ++i)
        {
            uint32_t filters = filtersList[i];
            discOut = (new Conv2D(filters, 3, stride, Tensor::GetPadding(Same, 3)))->Call(discOut);
            discOut = (new BatchNormalization())->Call(discOut);
            discOut = (new Activation(new LeakyReLU(0.2f)))->Call(discOut);
        }

        auto xFlat = (new Flatten())->Call(discOut);
        auto x = (new Dense(2, new Softmax()))->Call(xFlat);

        // this is single patch processing model
        auto patchDisc = new Flow(patchInput->Outputs(), { x[0], xFlat[0] });

        // generate final model for processing all patches
        auto imgInput = new Input(imgShape);

        // generate patches
        vector<TensorLike*> patches;

        for (int y = 0; y < ::sqrt(nbPatches); ++y)
        for (int x = 0; x < ::sqrt(nbPatches); ++x)
        {
            static auto patchExtract = [=](const vector<TensorLike*>& inputNodes)->vector<TensorLike*> { return { sub_tensor2d(inputNodes[0], patchSize, patchSize, patchSize * x, patchSize * y) }; };

            auto patch = (new Lambda(patchExtract))->Call(imgInput->Outputs())[0];
            patches.push_back(patch);
        }

        vector<TensorLike*> xList;
        vector<TensorLike*> xFlatList;

        for (size_t i = 0; i < patches.size(); ++i)
        {
            auto output = patchDisc->Call(patches[i], "patch_disc_" + i);
            xList.push_back(output[0]);
            xFlatList.push_back(output[1]);
        }

        TensorLike* xMerged;

        if (xList.size() > 1)
            xMerged = (new Concatenate(WidthAxis))->Call(xList)[0];
        else
            xMerged = xList[0];

        if (useMiniBatchDiscrimination)
        {
            static auto minb_disc = [](const vector<TensorLike*>& inputNodes) -> vector<TensorLike*>
            {
                NameScope scope("mini_batch_discriminator");
                auto x = inputNodes[0]; // x will be of shape [5, 100, 1, ?], where '?' is batch size
                // for first argument of difference we need to convert x to [1, 5, 100, ?] with simple batch reshape
                auto d1 = batch_reshape(x, Shape(1, x->GetShape().Width(), x->GetShape().Height()));
                // for second argument of difference we need to convert x to [?, 5, 100, 1] with a transpose
                auto d2 = transpose(x, { _3Axis, _0Axis, _1Axis, _2Axis });
                // due to broadcasting, the end result of following difference will be of shape [?, 5, 100, ?]
                auto diffs = sub(d1, d2);
                // first summation will produce a tensor with shape [?, 1, 100, ?]
                auto abs_diffs = sum(abs(diffs), _1Axis);
                // second summation will produce a tensor with shape [1, 1, 100, ?] and lastly we need a simple flatten to get 100 to the beginning [100, 1, 1, ?]
                return { batch_flatten(sum(exp(negative(abs_diffs)), _0Axis)) };
            };

            TensorLike* xFlatMerged;

            if (xFlatList.size() > 1)
                xFlatMerged = (new Concatenate(WidthAxis))->Call(xFlatList)[0];
            else
                xFlatMerged = xFlatList[0];

            uint32_t numKernels = 100;
            uint32_t dimPerKernel = 5;

            auto x_mbd = (new Dense(numKernels * dimPerKernel))->UseBias(false)->Call(xFlatMerged)[0];
            x_mbd = (new Reshape(Shape(dimPerKernel, numKernels)))->Call(x_mbd)[0];
            x_mbd = (new Lambda(minb_disc))->Call(x_mbd)[0];
            xMerged = (new Concatenate(WidthAxis))->Call({ xMerged, x_mbd })[0];
        }

        auto xOut = (new Dense(2, new Softmax()))->Call(xMerged);

        auto model = new Flow(imgInput->Outputs(), xOut);
        model->Optimize(new Adam(0.0001f), new BinaryCrossEntropy());
        return model;
    }
};
