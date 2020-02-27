#include "SRGAN.h"

///
ModelBase* SRGAN::CreateGenerator(const Shape& lrShape, const Shape& hrShape, uint32_t filtersStart, size_t residualBlocksNum)
{
    auto residualBlock = [](TensorLike* x, uint32_t nbFilters)
    {
        auto d = (new Conv2D(nbFilters, 3, 2, Tensor::GetPadding(Same, 3), new ReLU()))->Call(x)[0];
        d = (new BatchNormalization())->SetMomentum(0.8f)->Call(d)[0];
        d = (new Conv2D(nbFilters, 3, 1, Tensor::GetPadding(Same, 3)))->Call(d)[0];
        d = (new BatchNormalization())->SetMomentum(0.8f)->Call(d)[0];
        d = (new Merge(SumMerge))->Call({ d, x })[0];
        return d;
    };

    auto upconvBlock = [](TensorLike* x)
    {
        x = (new UpSampling2D(2))->Call(x)[0];
        x = (new Conv2D(256, 3, 1, Tensor::GetPadding(Same, 3), new ReLU()))->Call(x)[0];
        return x;
    };

    auto imgLr = (new Input(lrShape))->Outputs()[0];

    // Pre-residual block
    auto c1 = (new Conv2D(64, 9, 1, Tensor::GetPadding(Same, 9), new ReLU()))->Call(imgLr)[0];

    // Propogate through residual blocks
    auto r = residualBlock(c1, filtersStart);
    for (size_t i = 0; i < residualBlocksNum - 1; ++i)
        r = residualBlock(r, filtersStart);

    // Post-residual block
    auto c2 = (new Conv2D(64, 3, 1, Tensor::GetPadding(Same, 3)))->Call(r)[0];
    c2 = (new BatchNormalization())->SetMomentum(0.8f)->Call(c2)[0];
    c2 = (new Merge(SumMerge))->Call({ c2, c1 })[0];

    // Upsampling
    auto u1 = upconvBlock(c2);
    auto u2 = upconvBlock(u1);

    // Generate high resolution output
    auto imgHr = (new Conv2D(lrShape.Depth(), 9, 1, Tensor::GetPadding(Same, 9), new Tanh()))->Call(u2)[0];

    auto model = new Flow({ imgLr }, { imgHr }, "gen");
    return model;
}

///
Neuro::ModelBase* SRGAN::CreateDiscriminator(const Shape& hrShape, uint32_t filtersStart)
{
    auto block = [](TensorLike* x, uint32_t nbFilters, uint32_t stride = 1, bool norm = true)
    {
        x = (new Conv2D(nbFilters, 3, stride, Tensor::GetPadding(Same, 3), new LeakyReLU(0.2f)))->Call(x)[0];
        if (norm)
            x = (new BatchNormalization())->SetMomentum(0.8f)->Call(x)[0];
        return x;
    };

    auto d0 = (new Input(hrShape))->Outputs()[0];

    auto d1 = block(d0, filtersStart, 1, false);
    auto d2 = block(d1, filtersStart, 2);
    auto d3 = block(d2, filtersStart * 2);
    auto d4 = block(d3, filtersStart * 2, 2);
    auto d5 = block(d4, filtersStart * 4);
    auto d6 = block(d5, filtersStart * 4, 2);
    auto d7 = block(d6, filtersStart * 8);
    auto d8 = block(d7, filtersStart * 8, 2);
    
    auto d9 = (new Dense(filtersStart * 16, new LeakyReLU(0.2f)))->Call(d8)[0];
    auto validity = (new Dense(1, new Sigmoid()))->Call(d9)[0];

    auto model = new Flow({ d0 }, { validity });
    return model;
}
