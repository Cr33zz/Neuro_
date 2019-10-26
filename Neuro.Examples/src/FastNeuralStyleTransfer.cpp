#include "FastNeuralStyleTransfer.h"

//////////////////////////////////////////////////////////////////////////
TensorLike* FastNeuralStyleTransfer::CreateTransformerNet(TensorLike* input, TensorLike* training)
{
    NameScope scope("transform_net");
    auto convLayer = [&](TensorLike* input, uint32_t filtersNum, uint32_t filterSize, uint32_t stride = 1, bool doUpSample = false, bool doNorm = true, bool doReLU = true)
    {
        static int id = 1;
        NameScope scope("conv_" + to_string(id++));
        if (doUpSample)
            input = upsample2d(input, 2);
        input = (new Conv2D(filtersNum, filterSize, stride, Tensor::GetPadding(Same, filterSize)))->Call(input)[0];
        if (doNorm)
            input = (new BatchNormalization())->Call(input, training)[0];
        if (doReLU)
            input = relu(input);
        return input;
    };

    auto residualBlock = [&](TensorLike* input, uint32_t filterSize)
    {
        static int id = 1;
        NameScope scope("residual_block_" + to_string(id++));
        auto x = convLayer(input, 128, filterSize, 1);
        return add(input, convLayer(x, 128, filterSize, 1, false, true, false));
    };

    auto conv1 = convLayer(input, 32, 9, 1);
    auto conv2 = convLayer(conv1, 64, 3, 2);
    auto conv3 = convLayer(conv2, 128, 3, 2);
    auto resid1 = residualBlock(conv3, 3);
    auto resid2 = residualBlock(resid1, 3);
    auto resid3 = residualBlock(resid2, 3);
    auto resid4 = residualBlock(resid3, 3);
    auto resid5 = residualBlock(resid4, 3);
    auto up1 = convLayer(resid5, 64, 3, 1, true);
    auto up2 = convLayer(up1, 32, 3, 1, true);
    auto up3 = convLayer(up2, 3, 9, 1, false, false, false);
    return multiply(tanh(up3), 150.f);
}
