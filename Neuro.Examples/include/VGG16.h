#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <numeric>

#include "Neuro.h"

using namespace std;
using namespace Neuro;

//https://github.com/keras-team/keras-applications/blob/master/keras_applications/vgg16.py
// weights can be downloaded from: https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5
class VGG16
{
public:
    void Run()
    {
        Tensor image = LoadImage("data/mug.jpg", 224, 224);
        image.Sub(Tensor({ 103.939f, 116.779f, 123.68f }, Shape(3)), image);
        
        Tensor::SetForcedOpMode(GPU);

        auto model = CreateModel();

        cout << model->Summary();

        model->LoadWeights("data/vgg16_weights_tf_dim_ordering_tf_kernels.h5");

        auto prediction = model->Predict(image)[0];

        cout << prediction->ArgMax(WidthAxis)(0) << " " << prediction->Max(WidthAxis)(0) * 100 << "%" <<  endl;
    }

    //////////////////////////////////////////////////////////////////////////
    static ModelBase* CreateModel()
    {
        auto model = new Sequential("vgg16");
        model->AddLayer(new Conv2D(Shape(3, 224, 224), 64, 3, 1, 1, new ReLU(), NHWC, "block1_conv1"));
        model->AddLayer(new Conv2D(64, 3, 1, 1, new ReLU(), NHWC, "block1_conv2"));
        model->AddLayer(new MaxPooling2D(2, 2, 0, NHWC, "block1_pool"));
        model->AddLayer(new Conv2D(128, 3, 1, 1, new ReLU(), NHWC, "block2_conv1"));
        model->AddLayer(new Conv2D(128, 3, 1, 1, new ReLU(), NHWC, "block2_conv2"));
        model->AddLayer(new MaxPooling2D(2, 2, 0, NHWC, "block2_pool"));
        model->AddLayer(new Conv2D(256, 3, 1, 1, new ReLU(), NHWC, "block3_conv1"));
        model->AddLayer(new Conv2D(256, 3, 1, 1, new ReLU(), NHWC, "block3_conv2"));
        model->AddLayer(new Conv2D(256, 3, 1, 1, new ReLU(), NHWC, "block3_conv3"));
        model->AddLayer(new MaxPooling2D(2, 2, 0, NHWC, "block3_pool"));
        model->AddLayer(new Conv2D(512, 3, 1, 1, new ReLU(), NHWC, "block4_conv1"));
        model->AddLayer(new Conv2D(512, 3, 1, 1, new ReLU(), NHWC, "block4_conv2"));
        model->AddLayer(new Conv2D(512, 3, 1, 1, new ReLU(), NHWC, "block4_conv3"));
        model->AddLayer(new MaxPooling2D(2, 2, 0, NHWC, "block4_pool"));
        model->AddLayer(new Conv2D(512, 3, 1, 1, new ReLU(), NHWC, "block5_conv1"));
        model->AddLayer(new Conv2D(512, 3, 1, 1, new ReLU(), NHWC, "block5_conv2"));
        model->AddLayer(new Conv2D(512, 3, 1, 1, new ReLU(), NHWC, "block5_conv3"));
        model->AddLayer(new MaxPooling2D(2, 2, 0, NHWC, "block5_pool"));
        model->AddLayer(new Flatten("flatten"));
        model->AddLayer(new Dense(4096, new ReLU(), "fc1"));
        model->AddLayer(new Dense(4096, new ReLU(), "fc2"));
        model->AddLayer(new Dense(1000, new Softmax(), "predictions"));
        return model;
    }
};
