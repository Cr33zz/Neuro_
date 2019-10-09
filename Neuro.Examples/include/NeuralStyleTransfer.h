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
        Tensor contentImage = LoadImage("data/content.jpg", IMAGE_WIDTH, IMAGE_HEIGHT);
        Tensor styleImage = LoadImage("data/style.jpg", IMAGE_WIDTH, IMAGE_HEIGHT);
        Tensor outputImage = contentImage;

        // add image preprocessing....

        auto vgg16Model = VGG16::CreateModel();
        vgg16Model->LoadWeights("data/vgg16_weights_tf_dim_ordering_tf_kernels.h5");
        vgg16Model->SetTrainable(false);

        vector<const TensorLike*> contentLayers = { vgg16Model->Layer("block5_conv2")->OutputsAt(-1)[0] };
        vector<const TensorLike*> styleOutputs = { vgg16Model->Layer("block1_conv1")->OutputsAt(-1)[0], 
                                                   vgg16Model->Layer("block2_conv1")->OutputsAt(-1)[0], 
                                                   vgg16Model->Layer("block3_conv1")->OutputsAt(-1)[0], 
                                                   vgg16Model->Layer("block4_conv1")->OutputsAt(-1)[0],
                                                   vgg16Model->Layer("block5_conv1")->OutputsAt(-1)[0] };

        auto contentImg = new Variable(contentImage, "content_image");
        contentImg->Trainable(false);
        auto styleImg = new Variable(styleImage, "style_image");
        styleImg->Trainable(false);
        auto genImg = new Variable(contentImage, "gen_image");

        auto input = concatenate({ contentImg, styleImg, genImg }, BatchAxis);



        auto model = Flow(vgg16Model->InputsAt(-1), {});
    }

    TensorLike* GramMatrix(TensorLike* x)
    {
        assert(x->GetShape().Batch() == 1);
        //assuming NHWC format
        auto features = flatten(x);
        return matmul(features, transpose(features), "gram_matrix");
    }

    TensorLike* StyleLoss(TensorLike* style, TensorLike* gen)
    {
        assert(style->GetShape().Batch() == 1);
        assert(gen->GetShape().Batch() == 1);

        auto s = GramMatrix(style);
        auto g = GramMatrix(gen);

        const float channels = 3;
        const float size = (float)(IMAGE_WIDTH * IMAGE_HEIGHT);

        return div(sum(square(sub(s, g))), new Constant(4.f * (channels * channels) * (size * size)), "style_loss");
    }

    TensorLike* ContentLoss(TensorLike* content, TensorLike* gen)
    {
        return sum(square(sub(gen, content)), GlobalAxis, "content_loss");
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
