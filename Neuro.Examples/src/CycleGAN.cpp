#include "CycleGAN.h"

//////////////////////////////////////////////////////////////////////////
ModelBase* CycleGAN::CreateGenerator(const Shape& imgShape, uint32_t filtersStart)
{
    auto encoderBlock = [](TensorLike* x, uint32_t nbFilters, uint32_t fSize = 4)
    {
        x = (new ZeroPadding2D(2, 1, 2, 1))->Call(x)[0];
        x = (new Conv2D(nbFilters, fSize, 2, 0, new LeakyReLU(0.2f)))->Call(x)[0];
        x = (new InstanceNormalization())->Call(x)[0];
        return x;
    };

    auto decoderBlock = [](TensorLike* x, TensorLike* x2, uint32_t nbFilters, uint32_t fSize = 4, float dropoutRate = 0)
    {
        x = (new UpSampling2D(2))->Call(x)[0];
        x = (new ZeroPadding2D(2, 1, 2, 1))->Call(x)[0];
        x = (new Conv2D(nbFilters, fSize, 1, 0, new ReLU()))->Call(x)[0];
        if (dropoutRate)
            x = (new Dropout(dropoutRate))->Call(x)[0];
        x = (new InstanceNormalization())->Call(x)[0];
        x = (new Concatenate(DepthAxis))->Call({ x, x2 })[0];
        return x;
    };

    auto d0 = (new Input(imgShape))->Outputs()[0];

    // Encoder
    auto d1 = encoderBlock(d0, filtersStart);
    auto d2 = encoderBlock(d1, filtersStart * 2);
    auto d3 = encoderBlock(d2, filtersStart * 4);
    auto d4 = encoderBlock(d3, filtersStart * 8);

    // Decoder
    auto u1 = decoderBlock(d4, d3, filtersStart * 4);
    auto u2 = decoderBlock(u1, d2, filtersStart * 2);
    auto u3 = decoderBlock(u2, d1, filtersStart);

    auto u4 = (new UpSampling2D(2))->Call(u3)[0];
    auto u5 = (new ZeroPadding2D(2, 1, 2, 1))->Call(u4)[0];

    auto output = (new Conv2D(imgShape.Depth(), 4, 1, 0, new Tanh()))->Call(u5)[0];

    auto model = new Flow({ d0 }, { output }, "gen");
    return model;
}

//////////////////////////////////////////////////////////////////////////
ModelBase* CycleGAN::CreateDiscriminator(const Shape& imgShape, uint32_t filtersStart)
{
    auto block = [](TensorLike* x, uint32_t nbFilters, uint32_t fSize = 4, bool norm = true)
    {
        x = (new ZeroPadding2D(2, 1, 2, 1))->Call(x)[0];
        x = (new Conv2D(nbFilters, fSize, 2, 0, new LeakyReLU(0.2f)))->Call(x)[0];
        if (norm)
            x = (new InstanceNormalization())->Call(x)[0];
        return x;
    };

    auto d0 = (new Input(imgShape))->Outputs()[0];

    auto d1 = block(d0, filtersStart, 4, false);
    auto d2 = block(d1, filtersStart * 2);
    auto d3 = block(d2, filtersStart * 4);
    auto d4 = block(d3, filtersStart * 8);

    auto d5 = (new ZeroPadding2D(2, 1, 2, 1))->Call(d4)[0];
    auto output = (new Conv2D(1, 4, 1, 0, new Sigmoid()))->Call(d5)[0];

    auto model = new Flow({ d0 }, { output });
    return model;
}