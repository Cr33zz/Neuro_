#include "FastNeuralStyleTransfer.h"

//////////////////////////////////////////////////////////////////////////
TensorLike* FastNeuralStyleTransfer::CreateTransformerNet(TensorLike* input, TensorLike* training)
{
    auto convLayer = [&](TensorLike* input, uint32_t filtersNum, uint32_t filterSize, uint32_t stride, bool includeReLU = true)
    {
        input = (new Conv2D(filtersNum, filterSize, stride, Tensor::GetPadding(Same, filterSize)))->Call(input)[0];
        input = (new InstanceNormalization())->Call(input, training)[0];
        if (includeReLU)
            input = relu(input);
        return input;
    };

    auto residualBlock = [&](TensorLike* input, uint32_t filterSize)
    {
        auto x = convLayer(input, 128, filterSize, 1);
        return add(input, convLayer(x, 128, filterSize, 1, false));
    };

    auto upsampleLayer = [&](TensorLike* input, uint32_t filtersNum, uint32_t filterSize, uint32_t stride, uint32_t upsampleFactor)
    {
        input = upsample2d(input, upsampleFactor);
        input = (new Conv2D(filtersNum, filterSize, stride, Tensor::GetPadding(Same, filterSize)))->Call(input)[0];
        return input;
    };

    auto conv1 = convLayer(input, 32, 9, 1);
    auto conv2 = convLayer(conv1, 64, 3, 2);
    auto conv3 = convLayer(conv2, 128, 3, 2);
    auto resid1 = residualBlock(conv3, 3);
    auto resid2 = residualBlock(resid1, 3);
    auto resid3 = residualBlock(resid2, 3);
    auto resid4 = residualBlock(resid3, 3);
    auto resid5 = residualBlock(resid4, 3);
    auto up1 = upsampleLayer(resid5, 64, 3, 1, 2);
    auto up2 = upsampleLayer(up1, 32, 3, 1, 2);
    auto up3 = convLayer(up2, 3, 9, 1, false);
    return add(multiply(tanh(up3), 127.5f), 127.5f, "denormalized");
}
