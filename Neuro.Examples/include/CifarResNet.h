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

    LayerBase* ResNetLayer(LayerBase* inputLayer, size_t filtersNum = 16, size_t kernelSize = 3, size_t strides = 1, const ActivationBase& activation = ReLU(), bool batchNorm = true, bool convFirst = true)
    {
        //auto conv = new Conv2D()
    }
};

