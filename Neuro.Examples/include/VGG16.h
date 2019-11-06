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
        Tensor::SetForcedOpMode(GPU);

        Tensor image = LoadImage("data/mug.jpg", 224, 224);
        PreprocessImage(image, NHWC);

        auto model = CreateModel(NHWC);

        cout << model->Summary();

        auto prediction = model->Predict(image)[0];

        cout << prediction->ArgMax(WidthAxis)(0) << " " << prediction->Max(WidthAxis)(0) * 100 << "%" <<  endl;
    }

    static ModelBase* CreateModel(EDataFormat dataFormat, Shape inputShape = Shape(), bool includeTop = true, EPoolingMode poolMode = MaxPool);
    static TensorLike* Preprocess(TensorLike* image, EDataFormat dataFormat);
    static TensorLike* Deprocess(TensorLike* image, EDataFormat dataFormat);

    static void PreprocessImage(Tensor& image, EDataFormat dataFormat);
    static void DeprocessImage(Tensor& image, EDataFormat dataFormat);
};
