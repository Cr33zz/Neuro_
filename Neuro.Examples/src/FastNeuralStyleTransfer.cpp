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
    auto up3 = convLayer(up2, 3, 9, 1, false, true, false);
    return multiply(tanh(up3), 150.f);
}

//////////////////////////////////////////////////////////////////////////
ModelBase* FastNeuralStyleTransfer::CreateGeneratorModel(uint32_t width, uint32_t height, Placeholder* training)
{
    NameScope scope("generator");

    auto residual_block = [&](TensorLike* x, int num)
    {
        auto shortcut = x;
        x = (new Conv2D(128, 3, 1, Tensor::GetPadding(Same, 3), nullptr, NCHW, "resi_conv_" + to_string(num) + "_1"))->Call(x)[0];
        x = (new BatchNormalization("resi_normal_" + to_string(num) + "_1"))->Call(x, training)[0];
        x = (new Activation(new ReLU(), "resi_relu_" + to_string(num) + "_1"))->Call(x)[0];
        x = (new Conv2D(128, 3, 1, Tensor::GetPadding(Same, 3), nullptr, NCHW, "resi_conv_" + to_string(num) + "_2"))->Call(x)[0];
        x = (new BatchNormalization("resi_normal_" + to_string(num) + "_2"))->Call(x, training)[0];
        auto m = (new Merge(MergeSum, nullptr, "resi_add_" + to_string(num)))->Call({ x, shortcut })[0];
        return m;
    };

    auto input_o = new Input(Shape(width, height, 3), "input_o");

    auto c1 = (new Conv2D(32, 9, 1, Tensor::GetPadding(Same, 9), nullptr, NCHW, "conv_1"))->Call(input_o->Outputs())[0];
    c1 = (new BatchNormalization("normal_1"))->Call(c1, training)[0];
    c1 = (new Activation(new ReLU(), "relu_1"))->Call(c1)[0];

    auto c2 = (new Conv2D(64, 3, 2, Tensor::GetPadding(Same, 3), nullptr, NCHW, "conv_2"))->Call(c1)[0];
    c2 = (new BatchNormalization("normal_2"))->Call(c2, training)[0];
    c2 = (new Activation(new ReLU(), "relu_2"))->Call(c2)[0];

    auto c3 = (new Conv2D(128, 3, 2, Tensor::GetPadding(Same, 3), nullptr, NCHW, "conv_3"))->Call(c2)[0];
    c3 = (new BatchNormalization("normal_3"))->Call(c3, training)[0];
    c3 = (new Activation(new ReLU(), "relu_3"))->Call(c3)[0];

    auto r1 = residual_block(c3, 1);
    auto r2 = residual_block(r1, 2);
    auto r3 = residual_block(r2, 3);
    auto r4 = residual_block(r3, 4);
    auto r5 = residual_block(r4, 5);

    auto d1 = (new UpSampling2D(2, "up_1"))->Call(r5)[0];
    d1 = (new Conv2D(64, 3, 1, Tensor::GetPadding(Same, 3), nullptr, NCHW, "conv_4"))->Call(d1)[0];
    d1 = (new BatchNormalization("normal_4"))->Call(d1, training)[0];
    d1 = (new Activation(new ReLU(), "relu_4"))->Call(d1)[0];

    auto d2 = (new UpSampling2D(2, "up_2"))->Call(d1)[0];
    d2 = (new Conv2D(32, 3, 1, Tensor::GetPadding(Same, 3), nullptr, NCHW, "conv_5"))->Call(d2)[0];
    d2 = (new BatchNormalization("normal_5"))->Call(d2, training)[0];
    d2 = (new Activation(new ReLU(), "relu_5"))->Call(d2)[0];

    auto c4 = (new Conv2D(3, 9, 1, Tensor::GetPadding(Same, 9), nullptr, NCHW, "conv_6"))->Call(d2)[0];
    c4 = (new BatchNormalization("normal_6"))->Call(c4, training)[0];
    c4 = (new Activation(new Tanh(), "tanh_1"))->Call(c4)[0];
    c4 = (new OutputScale("output"))->Call(c4)[0];

    return new Flow(input_o->Outputs(), { c4 }, "generator_model");
}
