#include "Applications/VGG16.h"
#include "Tensors/Tensor.h"
#include "Models/Sequential.h"
#include "Layers/Conv2D.h"
#include "Layers/Pooling2D.h"
#include "Layers/Dense.h"
#include "Layers/Flatten.h"
#include "ComputationalGraph/Constant.h"
#include "ComputationalGraph/Ops.h"
#include "Activations.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    void VGG16::PreprocessImage(Tensor& image, EDataFormat dataFormat, bool swapChannels)
    {
        image.Sub(Tensor({ 123.68f, 116.779f, 103.939f }, dataFormat == NHWC ? Shape(3) : Shape(1, 1, 3)), image);
        //VGG networks were trained on BGR images so for RGB we have to swap channels
        if (swapChannels) // weights in first conv layer of VGG are expecting image in BGR format
            SwapChannels(image);
    }

    //////////////////////////////////////////////////////////////////////////
    void VGG16::DeprocessImage(Tensor& image, EDataFormat dataFormat, bool swapChannels, bool clipValues)
    {
        image.Add(Tensor({ 123.68f, 116.779f, 103.939f }, dataFormat == NHWC ? Shape(3) : Shape(1, 1, 3)), image);
        // because VGG networks were trained on BGR images they generate feature maps in BGR "style"
        // if these feature maps are used to generate image, the result will also be in BGR and will need 
        // channels swap if user wants RGB
        if (swapChannels)
            SwapChannels(image);

        if (clipValues)
            image.Clipped(0, 255, image);
    }

    //////////////////////////////////////////////////////////////////////////
    void VGG16::SwapChannels(Tensor& image)
    {
        Tensor temp(image);
        for (uint32_t n = 0; n < image.Batch(); ++n)
        {
            temp.CopyDepthTo(0, n, 2, n, image);
            temp.CopyDepthTo(2, n, 0, n, image);
        }
    }

    //////////////////////////////////////////////////////////////////////////
    Neuro::ModelBase* VGG16::CreateModel(EDataFormat dataFormat, Shape inputShape, bool includeTop, EPoolingMode poolMode, const string& weightsDir)
    {
        if (!inputShape.IsValid())
            inputShape = dataFormat == NHWC ? Shape(3, 224, 224) : Shape(224, 224, 3);

        //NEURO_ASSERT(dataFormat == NHWC) // check number of channels

        auto model = new Sequential("vgg16");
        model->AddLayer(new Conv2D(inputShape, 64, 3, 1, 1, new ReLU(), dataFormat, "block1_conv1"));
        model->AddLayer(new Conv2D(64, 3, 1, 1, new ReLU(), dataFormat, "block1_conv2"));
        model->AddLayer(new Pooling2D(2, 2, 0, poolMode, dataFormat, "block1_pool"));
        model->AddLayer(new Conv2D(128, 3, 1, 1, new ReLU(), dataFormat, "block2_conv1"));
        model->AddLayer(new Conv2D(128, 3, 1, 1, new ReLU(), dataFormat, "block2_conv2"));
        model->AddLayer(new Pooling2D(2, 2, 0, poolMode, dataFormat, "block2_pool"));
        model->AddLayer(new Conv2D(256, 3, 1, 1, new ReLU(), dataFormat, "block3_conv1"));
        model->AddLayer(new Conv2D(256, 3, 1, 1, new ReLU(), dataFormat, "block3_conv2"));
        model->AddLayer(new Conv2D(256, 3, 1, 1, new ReLU(), dataFormat, "block3_conv3"));
        model->AddLayer(new Pooling2D(2, 2, 0, poolMode, dataFormat, "block3_pool"));
        model->AddLayer(new Conv2D(512, 3, 1, 1, new ReLU(), dataFormat, "block4_conv1"));
        model->AddLayer(new Conv2D(512, 3, 1, 1, new ReLU(), dataFormat, "block4_conv2"));
        model->AddLayer(new Conv2D(512, 3, 1, 1, new ReLU(), dataFormat, "block4_conv3"));
        model->AddLayer(new Pooling2D(2, 2, 0, poolMode, dataFormat, "block4_pool"));
        model->AddLayer(new Conv2D(512, 3, 1, 1, new ReLU(), dataFormat, "block5_conv1"));
        model->AddLayer(new Conv2D(512, 3, 1, 1, new ReLU(), dataFormat, "block5_conv2"));
        model->AddLayer(new Conv2D(512, 3, 1, 1, new ReLU(), dataFormat, "block5_conv3"));
        model->AddLayer(new Pooling2D(2, 2, 0, poolMode, dataFormat, "block5_pool"));

        if (includeTop)
        {
            model->AddLayer(new Flatten("flatten"));
            model->AddLayer(new Dense(4096, new ReLU(), "fc1"));
            model->AddLayer(new Dense(4096, new ReLU(), "fc2"));
            model->AddLayer(new Dense(1000, new Softmax(), "predictions"));
        }

        if (includeTop)
            model->LoadWeights(weightsDir + "vgg16_weights_tf_dim_ordering_tf_kernels.h5");
        else
            model->LoadWeights(weightsDir + "vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5");

        return model;
    }

    //////////////////////////////////////////////////////////////////////////
    TensorLike* VGG16::Preprocess(TensorLike* image, EDataFormat dataFormat, bool swapChannels)
    {
        NameScope scope("vgg_preprocess");
        if (swapChannels)
        {
            image = swap_red_blue_channels(image);
            return sub(image, new Constant(Tensor({ 103.939f, 116.779f, 123.68f }, dataFormat == NHWC ? Shape(3) : Shape(1, 1, 3)), "mean_RGB"));
        }

        return sub(image, new Constant(Tensor({ 123.68f, 116.779f, 103.939f }, dataFormat == NHWC ? Shape(3) : Shape(1, 1, 3)), "mean_RGB"));
    }

    //////////////////////////////////////////////////////////////////////////
    TensorLike* VGG16::Deprocess(TensorLike* image, EDataFormat dataFormat, bool swapChannels, bool clipValues)
    {
        NameScope scope("vgg_deprocess");
        if (swapChannels)
        {
            image = add(image, new Constant(Tensor({ 103.939f, 116.779f, 123.68f }, dataFormat == NHWC ? Shape(3) : Shape(1, 1, 3)), "mean_RGB"));
            image = swap_red_blue_channels(image);
        }
        else
        {
            image = add(image, new Constant(Tensor({ 123.68f, 116.779f, 103.939f }, dataFormat == NHWC ? Shape(3) : Shape(1, 1, 3)), "mean_RGB"));
        }
        return clipValues ? clip(image, 0, 255) : image;
    }
}