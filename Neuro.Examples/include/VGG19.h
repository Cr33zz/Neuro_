#pragma once

#include "VGG16.h"

//https://github.com/keras-team/keras-applications/blob/master/keras_applications/vgg19.py
// weights can be downloaded from: 
//https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels.h5
//https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5
class VGG19 : public VGG16
{
public:
    void Run()
    {
        Tensor::SetForcedOpMode(GPU);

        Tensor image = LoadImage("data/mug.jpg", 224, 224);
        PreprocessImage(image, NHWC);

        auto model = CreateModel(NHWC);

        cout << model->Summary();

        model->LoadWeights("data/vgg16_weights_tf_dim_ordering_tf_kernels.h5");

        auto prediction = model->Predict(image)[0];

        cout << prediction->ArgMax(WidthAxis)(0) << " " << prediction->Max(WidthAxis)(0) * 100 << "%" << endl;
    }

    static ModelBase* CreateModel(EDataFormat dataFormat, Shape inputShape = Shape(), bool includeTop = true);
};
