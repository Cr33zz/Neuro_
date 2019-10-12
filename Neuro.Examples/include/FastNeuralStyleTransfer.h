#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <numeric>
#include <experimental/filesystem>

#include "Neuro.h"
#include "VGG19.h"

using namespace std;
using namespace Neuro;
namespace fs = std::experimental::filesystem;

class FastNeuralStyleTransfer
{
public:
    void Run()
    {
        const uint32_t IMAGE_WIDTH = 256;
        const uint32_t IMAGE_HEIGHT = 256;

        const float CONTENT_WEIGHT = 7.5e0f;
        const float STYLE_WEIGHT = 1e2f;

        const float LEARNING_RATE = 1e-3f;
        const int NUM_EPOCHS = 2;
        const uint32_t BATCH_SIZE = 4;

        const string STYLE_FILE = "data/style.jpg";
        const string CONTENT_FILES_DIR = "e:/Downloads/coco14";

        Tensor::SetForcedOpMode(GPU);

        // build content files list
        vector<string> contentFiles;
        for (const auto & entry : fs::directory_iterator(CONTENT_FILES_DIR))
            contentFiles.push_back(entry.path().generic_string());

        auto vggModel = VGG16::CreateModel(NCHW, Shape(IMAGE_WIDTH, IMAGE_HEIGHT, 3), false);
        vggModel->SetTrainable(false);

        vector<TensorLike*> contentOutputs = { vggModel->Layer("block4_conv2")->Outputs()[0] };
        vector<TensorLike*> styleOutputs = { vggModel->Layer("block1_conv1")->Outputs()[0],
                                             vggModel->Layer("block2_conv1")->Outputs()[0],
                                             vggModel->Layer("block3_conv1")->Outputs()[0],
                                             vggModel->Layer("block4_conv1")->Outputs()[0],
                                             vggModel->Layer("block5_conv1")->Outputs()[0] };

        auto vggContentFeaturesModel = Flow(vggModel->InputsAt(-1), contentOutputs);
        auto vggFeaturesModel = Flow(vggModel->InputsAt(-1), MergeVectors({ contentOutputs, styleOutputs }));
        
        // pre-compute style features of style image (we only need to do it once since that image won't change either)
        Tensor styleImage = LoadImage(STYLE_FILE, IMAGE_WIDTH, IMAGE_HEIGHT, NCHW);
        styleImage.SaveAsImage("_style.jpg", false);
        VGG16::PreprocessImage(styleImage, NCHW);

        auto styleFeatures = vggFeaturesModel.Eval(styleOutputs, { { (Placeholder*)(vggFeaturesModel.Inputs()[0]), &styleImage } });
        vector<Constant*> styleGrams;
        for (size_t i = 0; i < styleFeatures.size(); ++i)
        {
            Tensor* x = styleFeatures[i];
            uint32_t elementsPerFeature = x->GetShape().Width() * x->GetShape().Height();
            auto features = x->Reshaped(Shape(elementsPerFeature, x->GetShape().Depth()));
            styleGrams.push_back(new Constant(features.Mul(features.Transposed()).Mul(1.f / (float)elementsPerFeature), "style_" + to_string(i) + "_gram"));
        }

        // generate final computational graph
        auto content = new Placeholder(Shape(IMAGE_WIDTH, IMAGE_HEIGHT, 3), "content");
        auto contentPre = VGG16::Preprocess(content, NCHW);

        auto targetContentFeatures = vggContentFeaturesModel(contentPre);

        //auto outputImg = new Variable(contentImage, "output_image");
        auto stylizedContent = CreateTransformerNet(multiply(content, 1.f / 255.f), new Constant(1.f)); // normalize input images
        auto stylizedContentPre = VGG16::Preprocess(stylizedContent, NCHW);

        auto stylizedFeatures = vggFeaturesModel(stylizedContentPre);

        float contentLossWeight = 1e3f;
        float styleLossWeight = 1e-2f;

        // compute content loss from first output...
        auto contentLoss = multiply(ContentLoss(targetContentFeatures[0], stylizedFeatures[0]), contentLossWeight);
        stylizedFeatures.erase(stylizedFeatures.begin());

        vector<TensorLike*> styleLosses;
        // ... and style losses from remaining outputs
        assert(stylizedFeatures.size() == styleGrams.size());
        for (size_t i = 0; i < stylizedFeatures.size(); ++i)
            styleLosses.push_back(StyleLoss(styleGrams[i], stylizedFeatures[i], (int)i));
        auto styleLoss = multiply(merge_avg(styleLosses, "mean_style_loss"), styleLossWeight, "style_loss");

        auto totalLoss = add(contentLoss, styleLoss, "total_loss");

        auto optimizer = Adam(5.f, 0.99f, 0.999f, 0.1f);
        auto minimize = optimizer.Minimize({ totalLoss });

        Tensor contentBatch(Shape::From(content->GetShape(), BATCH_SIZE));

        size_t samples = contentFiles.size();
        size_t steps = samples / BATCH_SIZE;

        for (int e = 1; e <= NUM_EPOCHS; ++e)
        {
            cout << "Epoch " << e << endl;

            Tqdm progress(steps, 10);
            progress.ShowStep(true).ShowElapsed(false);
            for (int i = 0; i < steps; ++i, progress.NextStep())
            {
                contentBatch.OverrideHost();
                //load images
                for (int j = 0; j < BATCH_SIZE; ++j)
                    LoadImage(contentFiles[i * BATCH_SIZE + j], &contentBatch.GetValues()[0] + j * contentBatch.BatchLength(), content->GetShape().Width(), content->GetShape().Height(), NCHW);
        
                auto results = Session::Default()->Run({ stylizedContent, contentLoss, styleLoss, totalLoss, minimize }, { { content, &contentBatch } });

                stringstream extString;
                extString << setprecision(4) << fixed << " - content_l: " << (*results[1])(0) << " - style_l: " << (*results[2])(0) << " - total_l: " << (*results[3])(0);
                progress.SetExtraString(extString.str());

                if (i % 10 == 0)
                {
                    auto genImage = *results[0];
                    genImage.SaveAsImage("fnst_e" + PadLeft(to_string(e), 4, '0') + "_b" + PadLeft(to_string(i), 4, '0') + ".png", false);
                }
            }
        }

        //auto results = Session::Default()->Run({ outputImg }, {});
        //auto genImage = *results[0];
        //VGG16::UnprocessImage(genImage, NCHW);
        //genImage.SaveAsImage("_neural_transfer.jpg", false);
    }

    //////////////////////////////////////////////////////////////////////////
    TensorLike* CreateTransformerNet(TensorLike* input, TensorLike* training)
    {
        auto convLayer = [&](TensorLike* input, uint32_t filtersNum, uint32_t filterSize, uint32_t stride, bool includeReLU = true)
        {
            input = (new Conv2D(filtersNum, filterSize, stride, Tensor::GetPadding(Same, filterSize)))->Call(input)[0];
            input = (new InstanceNormalization())->Call(input, training)[0];
            if (includeReLU)
                input = relu(input);
            return input;
        };

        auto residualBlock = [&](TensorLike* input, uint32_t filterSize)
        {
            auto x = convLayer(input, 128, filterSize, 1);
            return add(input, convLayer(x, 128, filterSize, 1, false));
        };

        auto upsampleLayer = [&](TensorLike* input, uint32_t filtersNum, uint32_t filterSize, uint32_t stride, uint32_t upsampleFactor)
        {
            input = upsample2d(input, upsampleFactor);
            input = (new Conv2D(filtersNum, filterSize, stride, Tensor::GetPadding(Same, filterSize)))->Call(input)[0];
            return input;
        };

        auto conv1 = convLayer(input, 32, 9, 1);
        auto conv2 = convLayer(conv1, 64, 3, 2);
        auto conv3 = convLayer(conv2, 128, 3, 2);
        auto resid1 = residualBlock(conv3, 3);
        auto resid2 = residualBlock(resid1, 3);
        auto resid3 = residualBlock(resid2, 3);
        auto resid4 = residualBlock(resid3, 3);
        auto resid5 = residualBlock(resid4, 3);
        auto up1 = upsampleLayer(resid5, 64, 3, 2, 2);
        auto up2 = upsampleLayer(up1, 32, 3, 2, 2);
        auto up3 = convLayer(up2, 3, 9, 1, false);
        return add(multiply(tanh(up3), 127.5f), 127.5f); // de-normalize
    }

    //////////////////////////////////////////////////////////////////////////
    TensorLike* GramMatrix(TensorLike* x, const string& name)
    {
        assert(x->GetShape().Batch() == 1);

        uint32_t elementsPerFeature = x->GetShape().Width() * x->GetShape().Height();
        auto features = reshape(x, Shape(elementsPerFeature, x->GetShape().Depth()));
        return multiply(matmul(features, transpose(features)), 1.f / (float)elementsPerFeature, name + "_gram_matrix");
        //return matmul(features, transpose(features));
    }

    //////////////////////////////////////////////////////////////////////////
    TensorLike* StyleLoss(TensorLike* styleGram, TensorLike* gen, int index)
    {
        assert(gen->GetShape().Batch() == 1);

        //auto s = GramMatrix(style, index);
        auto genGram = GramMatrix(gen, "gen_style_" + to_string(index));

        float channels = (float)gen->GetShape().Depth();
        float size = (float)(gen->GetShape().Height() * gen->GetShape().Width());

        //return multiply(mean(square(sub(styleGram, genGram))), 1.f / (4.f * (channels * channels) * (size * size)), "style_loss_" + to_string(index));
        //return div(mean(square(sub(styleGram, genGram))), new Constant(4.f * (channels * channels) * (size * size)), "style_loss_" + to_string(index));
        return mean(square(sub(styleGram, genGram)));
    }

    //////////////////////////////////////////////////////////////////////////
    TensorLike* ContentLoss(TensorLike* target, TensorLike* gen)
    {
        return mean(square(sub(target, gen)), GlobalAxis, "content_loss");
    }
};
