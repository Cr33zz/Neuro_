#pragma once

#include "Applications/VGG16.h"

namespace Neuro
{
    //https://github.com/keras-team/keras-applications/blob/master/keras_applications/vgg19.py
    // weights can be downloaded from: 
    //https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels.h5
    //https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5
    struct NEURO_DLL_EXPORT VGG19
    {
        static ModelBase* CreateModel(EDataFormat dataFormat, Shape inputShape = Shape(), bool includeTop = true, EPoolingMode poolMode = MaxPool, const string& weightsDir = "");
    };
}