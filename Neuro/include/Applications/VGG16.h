#pragma once

#include "Types.h"
#include "Tensors/Shape.h"

namespace Neuro
{
    class ModelBase;
    class Tensor;
    class TensorLike;

    //https://github.com/keras-team/keras-applications/blob/master/keras_applications/vgg16.py
    // weights can be downloaded from: https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5
    struct NEURO_DLL_EXPORT VGG16
    {
        static ModelBase* CreateModel(EDataFormat dataFormat, Shape inputShape = Shape(), bool includeTop = true, EPoolingMode poolMode = MaxPool, const string& weightsDir = "");
        static TensorLike* Preprocess(TensorLike* image, EDataFormat dataFormat, bool swapChannels = true);
        static TensorLike* Deprocess(TensorLike* image, EDataFormat dataFormat, bool swapChannels = true, bool clipValues = true);

        static void PreprocessImage(Tensor& image, EDataFormat dataFormat, bool swapChannels = true);
        static void DeprocessImage(Tensor& image, EDataFormat dataFormat, bool swapChannels = true, bool clipValues = true);

        static Tensor PreprocessImageCopy(const Tensor& image, EDataFormat dataFormat, bool swapChannels = true);
        static Tensor DeprocessImageCopy(const Tensor& image, EDataFormat dataFormat, bool swapChannels = true, bool clipValues = true);

        static void SwapChannels(Tensor& image);
    };
}
