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
        
        Tensor contentImage = LoadImage("data/content.jpg", IMAGE_WIDTH, IMAGE_HEIGHT);
        VGG16::PreprocessImage(contentImage);
        Tensor styleImage = LoadImage("data/style.jpg", IMAGE_WIDTH, IMAGE_HEIGHT);
        VGG16::PreprocessImage(styleImage);
        
        auto vgg16Model = VGG16::CreateModel();
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

        // generate beginning of the computational graph for processing output image
        auto outputs = model(outputImg);

        // compute content loss from first output...
        auto contentLoss = ContentLoss(content, outputs[0]);
        outputs.erase(outputs.begin());

        vector<TensorLike*> styleLosses;
        // ... and style losses from remaining outputs
        assert(outputs.size() == styles.size());
        for (size_t i = 0; i < outputs.size(); ++i)
            styleLosses.push_back(StyleLoss(styles[i], outputs[i], i));

        auto totalLoss = add(contentLoss, merge_avg(styleLosses, "mean_style_loss"), "total_loss");

        auto optimizer = Adam(5.f, 0.99f, 0.999f, 0.1f);

        auto minimize = optimizer.Minimize({ totalLoss }, { outputImg });

        Debug::LogAllOutputs();

        const int EPOCHS = 100;
        Tqdm progress(EPOCHS, 0);
        for (int e = 1; e < EPOCHS; ++e, progress.NextStep())
        {
            auto results = Session::Default()->Run({ totalLoss, outputImg, minimize }, {});
            progress.SetExtraString(" - loss: " + to_string((*results[0])(0)));

            auto genImage = *results[1];
            VGG16::UnprocessImage(genImage);
            genImage.SaveAsImage("neural_transfer_" + to_string(e) + ".png", false);
        }
    }

    TensorLike* GramMatrix(TensorLike* x, int index)
    {
        NameScope scope(x->Name() + "_gram_matrix" + to_string(index));
        assert(x->GetShape().Batch() == 1);

        //assuming NHWC format
        auto features = flatten(x);
        return matmul(features, transpose(features));
    }

    TensorLike* StyleLoss(TensorLike* style, TensorLike* gen, int index)
    {
        NameScope scope("style_loss" + to_string(index));
        assert(style->GetShape().Batch() == 1);
        assert(gen->GetShape().Batch() == 1);

        auto s = GramMatrix(style, index);
        auto g = GramMatrix(gen, index);

        const float channels = 3;
        const float size = (float)(IMAGE_WIDTH * IMAGE_HEIGHT);

        return multiply(mean(square(sub(s, g))), new Constant(1.f / (4.f * (channels * channels) * (size * size))));
    }

    TensorLike* ContentLoss(TensorLike* content, TensorLike* gen)
    {
        NameScope scope("content_loss");
        return mean(square(sub(gen, content)));
    }

    //class NeuralStyleLoss : public LossBase
    //{
    //    virtual LossBase* Clone() const override { return new NeuralStyleLoss(); }

    //    virtual void Compute(const Tensor& targetOutput, const Tensor& output, Tensor& result) override
    //    {
    //        // content loss
    //        mean()
    //    }

    //    virtual void Derivative(const Tensor& targetOutput, const Tensor& output, Tensor& result) const override
    //    {

    //    }
    //};
};
