#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <numeric>

#include "Neuro.h"
#include "VGG16.h"

using namespace std;
using namespace Neuro;

const size_t IMAGE_WIDTH = 224;
const size_t IMAGE_HEIGHT = 224;

//https://medium.com/tensorflow/neural-style-transfer-creating-art-with-deep-learning-using-tf-keras-and-eager-execution-7d541ac31398
class NeuralStyleTransfer
{
public:
    void Run()
    {
        Tensor::SetForcedOpMode(GPU);
        
        Tensor contentImage = LoadImage("data/content.jpg", IMAGE_WIDTH, IMAGE_HEIGHT, NCHW);
        VGG16::PreprocessImage(contentImage, NCHW);
        Tensor styleImage = LoadImage("data/style.jpg", IMAGE_WIDTH, IMAGE_HEIGHT, NCHW);
        VGG16::PreprocessImage(styleImage, NCHW);
        
        auto vgg16Model = VGG16::CreateModel(NCHW);
        vgg16Model->LoadWeights("data/vgg16_weights_tf_dim_ordering_tf_kernels.h5");
        vgg16Model->SetTrainable(false);

        vector<TensorLike*> contentOutputs = { vgg16Model->Layer("block5_conv2")->Outputs()[0] };
        vector<TensorLike*> styleOutputs = { vgg16Model->Layer("block1_conv1")->Outputs()[0], 
                                             vgg16Model->Layer("block2_conv1")->Outputs()[0], 
                                             vgg16Model->Layer("block3_conv1")->Outputs()[0], 
                                             vgg16Model->Layer("block4_conv1")->Outputs()[0],
                                             vgg16Model->Layer("block5_conv1")->Outputs()[0] };

        auto outputImg = new Variable(contentImage, "output_image");

        auto model = Flow(vgg16Model->InputsAt(-1), MergeVectors({ contentOutputs, styleOutputs }));

        // pre-compute content features of content image (we only need to do it once since that image won't change)
        auto contentFeatures = model.Predict(contentImage)[0];
        Constant* content = new Constant(*contentFeatures, "content");

        // pre-compute style features of style image (we only need to do it once since that image won't change either)
        auto styleFeatures = model.Predict(styleImage);
        styleFeatures.erase(styleFeatures.begin()); //get rid of content feature
        vector<Constant*> styles;
        for (size_t i = 0; i < styleFeatures.size(); ++i)
            styles.push_back(new Constant(*styleFeatures[i], "style_" + to_string(i)));
        vector<TensorLike*> styleGrams;
        for (size_t i = 0; i < styleFeatures.size(); ++i)
            styleGrams.push_back(GramMatrix(styles[i], "style_" + to_string(i)));

        // generate beginning of the computational graph for processing output image
        auto outputs = model(outputImg);

        // compute content loss from first output...
        auto contentLoss = ContentLoss(content, outputs[0]);
        outputs.erase(outputs.begin());

        vector<TensorLike*> styleLosses;
        // ... and style losses from remaining outputs
        assert(outputs.size() == styles.size());
        for (size_t i = 0; i < outputs.size(); ++i)
            styleLosses.push_back(StyleLoss(styleGrams[i], outputs[i], (int)i));

        auto totalLoss = add(contentLoss, merge_avg(styleLosses, "mean_style_loss"), "total_loss");

        auto optimizer = Adam(5.f, 0.99f, 0.999f, 0.1f);

        auto minimize = optimizer.Minimize({ totalLoss }, { outputImg });

        /*Debug::LogOutput("content_loss");
        Debug::LogOutput("style_loss");
        Debug::LogOutput("total_loss");
        Debug::LogOutput("mean_style_loss");*/

        const int EPOCHS = 1000;
        Tqdm progress(EPOCHS, 0);
        for (int e = 1; e < EPOCHS; ++e, progress.NextStep())
        {
            auto results = Session::Default()->Run({ totalLoss, outputImg, minimize }, {});
            progress.SetExtraString(" - loss: " + to_string((*results[0])(0)));

            if (e % 10 == 0)
            {
                auto genImage = *results[1];
                VGG16::UnprocessImage(genImage, NCHW);
                genImage.SaveAsImage("neural_transfer_" + to_string(e) + ".png", false);
            }
        }
    }

    //////////////////////////////////////////////////////////////////////////
    TensorLike* GramMatrix(TensorLike* x, const string& name)
    {
        assert(x->GetShape().Batch() == 1);

        auto features = reshape(x, Shape(x->GetShape().Width() * x->GetShape().Height(), x->GetShape().Depth()));
        return matmul(features, transpose(features), name + "_gram_matrix");
    }

    //////////////////////////////////////////////////////////////////////////
    TensorLike* StyleLoss(TensorLike* styleGram, TensorLike* gen, int index)
    {
        assert(gen->GetShape().Batch() == 1);

        //auto s = GramMatrix(style, index);
        auto genGram = GramMatrix(gen, "gen_style_" + to_string(index));

        const float channels = 3;
        const float size = (float)(IMAGE_WIDTH * IMAGE_HEIGHT);

        return div(mean(square(sub(styleGram, genGram))), new Constant(4.f * (channels * channels) * (size * size)), "style_loss_" + to_string(index));
        //return mean(square(sub(styleGram, genGram)), GlobalAxis);
    }

    //////////////////////////////////////////////////////////////////////////
    TensorLike* ContentLoss(TensorLike* content, TensorLike* gen)
    {
        NameScope scope("content_loss");
        return mean(square(sub(gen, content)));
    }
};
