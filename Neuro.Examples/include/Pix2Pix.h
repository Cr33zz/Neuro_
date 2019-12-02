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

        const Shape IMG_SHAPE(256, 256, 3);
        const Shape PATCH_SHAPE(64, 64, 3);
        const size_t PATCHES_NUM = 16;
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
        auto dModel = CreatePatchDiscriminator(PATCH_SHAPE, PATCHES_NUM);
        cout << "Discriminator" << endl << dModel->Summary();

        auto inSrc = new Input(IMG_SHAPE);
        auto genOut = gModel->Call(inSrc->Outputs());

        // generate patches
        vector<TensorLike*> patches;

        for (int y = 0; y < ::sqrt(PATCHES_NUM); ++y)
        for (int x = 0; x < ::sqrt(PATCHES_NUM); ++x)
            patches.push_back(sub_tensor(inSrc->Outputs()[0], PATCH_SHAPE.Width() * x, PATCH_SHAPE.Height() * y));

        auto disOut = dModel->Call(patches);

        auto ganModel = new Flow(inSrc->Outputs(), { disOut[0], genOut[0] }, "pix2pix");
        ganModel->Optimize(new Adam(0.0001f), { new BinaryCrossEntropy(), new MeanAbsoluteError() }, { 1.f, 100.f });
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
    
    ModelBase* CreatePatchDiscriminator(const Shape& patchShape, size_t nbPatches, bool useMiniBatchDiscrimination = true)
    {
        NEURO_ASSERT(patchShape.Width() == patchShape.Height(), "");
        uint32_t stride = 2;
        auto inputLayer = new Input(patchShape);

        uint32_t filtersStart = 64;
        size_t nbConv = int(floor(::log(patchShape.Width()) / ::log(2)));
        vector<uint32_t> filtersList(nbConv);
        for (int i = 0; i < nbConv; ++i)
            filtersList[i] = filtersStart * (uint32_t)min(8, ::pow(2, i));

        auto discOut = (new Conv2D(filtersList[0], 3, stride, Tensor::GetPadding(Same, 3), new LeakyReLU(0.2f)))->Call(inputLayer->Outputs());

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
        auto patchGan = new Flow(inputLayer->Outputs(), { x[0], xFlat[0] });

        // generate final model for processing all patches
        vector<TensorLike*> inputList(nbPatches);
        for (size_t i = 0; i < nbPatches; ++i)
            inputList[i] = (new Input(patchShape, "patch_input_" + i))->Outputs()[0];

        vector<TensorLike*> xList;
        vector<TensorLike*> xFlatList;

        for (auto& patch : inputList)
        {
            auto output = patchGan->Call(patch);
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

            uint32_t num_kernels = 100;
            uint32_t dim_per_kernel = 5;

            auto x_mbd = (new Dense(num_kernels * dim_per_kernel))->UseBias(false)->Call(xFlatMerged)[0];
            x_mbd = (new Reshape(Shape(dim_per_kernel, num_kernels)))->Call(x_mbd)[0];
            x_mbd = (new Lambda(minb_disc))->Call(x_mbd)[0];
            xMerged = (new Concatenate(WidthAxis))->Call({ xMerged, x_mbd })[0];
        }

        auto xOut = (new Dense(2, new Softmax()))->Call(xMerged);

        auto model = new Flow(inputList, xOut);
        model->Optimize(new Adam(0.0001f), new BinaryCrossEntropy());
        return model;
    }
};
