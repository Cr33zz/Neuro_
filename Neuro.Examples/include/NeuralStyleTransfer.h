#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <numeric>

#include "Neuro.h"
#include "VGG16.h"

using namespace std;
using namespace Neuro;

//https://medium.com/tensorflow/neural-style-transfer-creating-art-with-deep-learning-using-tf-keras-and-eager-execution-7d541ac31398
class NeuralStyleTransfer
{
public:
    void Run()
    {
        Tensor contentImage = LoadImage("data/content.jpg", 224, 224);
        Tensor styleImage = LoadImage("data/style.jpg", 224, 224);
        Tensor outputImage = contentImage;

        auto vgg16Model = VGG16::CreateModel();
        vgg16Model->LoadWeights("data/vgg16_weights_tf_dim_ordering_tf_kernels.h5");

        vector<const LayerBase*> contentLayers = { vgg16Model->Layer("block5_conv2") };
        vector<const LayerBase*> styleLayers = { vgg16Model->Layer("block1_conv1"), vgg16Model->Layer("block2_conv1"), vgg16Model->Layer("block3_conv1"), vgg16Model->Layer("block4_conv1"), vgg16Model->Layer("block5_conv1") };

        //auto model = Flow({contentInput, styleInput})
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
