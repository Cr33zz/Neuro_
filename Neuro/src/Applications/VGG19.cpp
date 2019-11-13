#include "Applications/VGG19.h"
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
    ModelBase* VGG19::CreateModel(EDataFormat dataFormat, Shape inputShape, bool includeTop, EPoolingMode poolMode, const string& weightsDir)
    {
        if (!inputShape.IsValid())
            inputShape = dataFormat == NHWC ? Shape(3, 224, 224) : Shape(224, 224, 3);

        //NEURO_ASSERT(dataFormat == NHWC) // check number of channels

        auto model = new Sequential("vgg19");
        model->AddLayer(new Conv2D(inputShape, 64, 3, 1, 1, new ReLU(), dataFormat, "block1_conv1"));
        model->AddLayer(new Conv2D(64, 3, 1, 1, new ReLU(), dataFormat, "block1_conv2"));
        model->AddLayer(new Pooling2D(2, 2, 0, poolMode, dataFormat, "block1_pool"));
        model->AddLayer(new Conv2D(128, 3, 1, 1, new ReLU(), dataFormat, "block2_conv1"));
        model->AddLayer(new Conv2D(128, 3, 1, 1, new ReLU(), dataFormat, "block2_conv2"));
        model->AddLayer(new Pooling2D(2, 2, 0, poolMode, dataFormat, "block2_pool"));
        model->AddLayer(new Conv2D(256, 3, 1, 1, new ReLU(), dataFormat, "block3_conv1"));
        model->AddLayer(new Conv2D(256, 3, 1, 1, new ReLU(), dataFormat, "block3_conv2"));
        model->AddLayer(new Conv2D(256, 3, 1, 1, new ReLU(), dataFormat, "block3_conv3"));
        model->AddLayer(new Conv2D(256, 3, 1, 1, new ReLU(), dataFormat, "block3_conv4"));
        model->AddLayer(new Pooling2D(2, 2, 0, poolMode, dataFormat, "block3_pool"));
        model->AddLayer(new Conv2D(512, 3, 1, 1, new ReLU(), dataFormat, "block4_conv1"));
        model->AddLayer(new Conv2D(512, 3, 1, 1, new ReLU(), dataFormat, "block4_conv2"));
        model->AddLayer(new Conv2D(512, 3, 1, 1, new ReLU(), dataFormat, "block4_conv3"));
        model->AddLayer(new Conv2D(512, 3, 1, 1, new ReLU(), dataFormat, "block4_conv4"));
        model->AddLayer(new Pooling2D(2, 2, 0, poolMode, dataFormat, "block4_pool"));
        model->AddLayer(new Conv2D(512, 3, 1, 1, new ReLU(), dataFormat, "block5_conv1"));
        model->AddLayer(new Conv2D(512, 3, 1, 1, new ReLU(), dataFormat, "block5_conv2"));
        model->AddLayer(new Conv2D(512, 3, 1, 1, new ReLU(), dataFormat, "block5_conv3"));
        model->AddLayer(new Conv2D(512, 3, 1, 1, new ReLU(), dataFormat, "block5_conv4"));
        model->AddLayer(new Pooling2D(2, 2, 0, poolMode, dataFormat, "block5_pool"));
        if (includeTop)
        {
            model->AddLayer(new Flatten("flatten"));
            model->AddLayer(new Dense(4096, new ReLU(), "fc1"));
            model->AddLayer(new Dense(4096, new ReLU(), "fc2"));
            model->AddLayer(new Dense(1000, new Softmax(), "predictions"));
        }

        if (includeTop)
            model->LoadWeights(weightsDir + "vgg19_weights_tf_dim_ordering_tf_kernels.h5", false);
        else
            model->LoadWeights(weightsDir + "vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5", false);

        return model;
    }
}

