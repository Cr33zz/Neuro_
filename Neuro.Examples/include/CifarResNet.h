#pragma once

#include <iostream>
#include <string>
#include <vector>

#include "Neuro.h"

using namespace std;
using namespace Neuro;

//based on https://keras.io/examples/cifar10_resnet/
class CifarResNet
{
    void Run()
    {
        Tensor::SetDefaultOpMode(GPU);

        Tensor images, labels;
        LoadCifar10Data("data/cifar10_data.bin", images, labels, false);
    }

    LayerBase* ResNetLayer(LayerBase* inputLayer, size_t filtersNum = 16, size_t kernelSize = 3, size_t strides = 1, ActivationBase* activation = new ReLU(), bool batchNorm = true, bool convFirst = true)
    {
        LayerBase* x = inputLayer;
        
        auto conv = (new Conv2D(filtersNum, kernelSize, strides, Tensor::GetPadding(Same, 3)))->KernelInitializer(new HeNormal())/*->KernelRegularizer(L2)*/;

        if (convFirst)
        {
            x = conv->Link(x);
            if (batchNorm)
                x = new BatchNormalization(x);
            if (activation)
                x = new Activation(x, activation);
        }
        else
        {
            if (batchNorm)
                x = new BatchNormalization(x);
            if (activation)
                x = new Activation(x, activation);
            x = conv->Link(x);
        }

        return x;
    }

    ModelBase* ResNetV1(const Shape& inputShape, size_t depth, size_t classesNum = 10)
    {
        assert((depth - 2) % 6 == 0); // depth should be 6n+2 (eg 20, 32, 44)

        const int RES_BLOCKS_NUM = int((depth - 2) / 6);
        size_t filtersNum = 16;

        LayerBase* inputs = new Input(inputShape);
        LayerBase* x = ResNetLayer(inputs);
        LayerBase* y = nullptr;
        for (int stack = 0; stack < 3; ++stack)
        {
            for (int resBlock = 0; resBlock < RES_BLOCKS_NUM; ++resBlock)
            {
                size_t strides = 1;
                
                if (stack > 0 && resBlock == 0)  // first layer but not first stack
                    strides = 2;  // down-sample

                y = ResNetLayer(x, filtersNum, 3, strides);
                y = ResNetLayer(y, filtersNum, 3, 1, nullptr);
                        
                if (stack > 0 && resBlock == 0)  // first layer but not first stack
                    x = ResNetLayer(x, filtersNum, 1, strides, nullptr, false); // linear projection residual shortcut connection to match changed dims

                x = new Merge({ x, y }, MergeSum, new ReLU());
            }

            filtersNum *= 2;
        }

        x = new AvgPooling2D(x, 8);
        y = new Flatten(x);
        auto outputs = (new Dense(y, classesNum, new Softmax()))->WeightsInitializer(new HeNormal());

        return new Flow({ inputs }, { outputs });
    }
};

