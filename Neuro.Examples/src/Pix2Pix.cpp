#include "Pix2Pix.h"

//////////////////////////////////////////////////////////////////////////
ModelBase* Pix2Pix::CreateGenerator(const Shape& imgShape)
{
    auto encoderBlock = [](TensorLike* input, uint32_t filtersNum, bool batchNorm = true)
    {
        //auto g = (new ZeroPadding2D(1, 1, 1, 1))->Call(input);
        auto g = (new Conv2D(filtersNum, 3, 2, Tensor::GetPadding(Same, 3)))->Call(input);
        if (batchNorm)
            g = (new BatchNormalization())->Call(g);
        g = (new Activation(new LeakyReLU(0.2f)))->Call(g);
        return g;
    };

    auto decoderBlock = [](TensorLike* input, TensorLike* skipInput, uint32_t filtersNum, bool dropout = true)
    {
        auto g = (new UpSampling2D(2))->Call(input);
        //g = (new ZeroPadding2D(2, 2, 2, 2))->Call(g);
        g = (new Conv2D(filtersNum, 3, 1, Tensor::GetPadding(Same, 3)))->Call(g);
        g = (new BatchNormalization())->Call(g);
        if (dropout)
            g = (new Dropout(0.5f))->Call(g);
        g = (new Concatenate(DepthAxis))->Call({ g[0], skipInput });
        g = (new Activation(new ReLU()))->Call(g);
        return g;
    };

    auto inImage = new Input(imgShape);

    ///encoder
    auto e1 = encoderBlock(inImage->Outputs()[0], 64, false);
    auto e2 = encoderBlock(e1[0], 128);
    auto e3 = encoderBlock(e2[0], 256);
    auto e4 = encoderBlock(e3[0], 512);
    auto e5 = encoderBlock(e4[0], 512);
    auto e6 = encoderBlock(e5[0], 512);
    auto e7 = encoderBlock(e6[0], 512);
    /// bottleneck
    //auto b = (new ZeroPadding2D(2, 2, 2, 2))->Call(e7);
    auto b = (new Conv2D(512, 3, 2, Tensor::GetPadding(Same, 3)))->Call(e7);
    b = (new Activation(new ReLU()))->Call(b);
    /// decoder
    auto d1 = decoderBlock(b[0], e7[0], 512);
    auto d2 = decoderBlock(d1[0], e6[0], 512);
    auto d3 = decoderBlock(d2[0], e5[0], 512);
    auto d4 = decoderBlock(d3[0], e4[0], 512, false);
    auto d5 = decoderBlock(d4[0], e3[0], 256, false);
    auto d6 = decoderBlock(d5[0], e2[0], 128, false);
    auto d7 = decoderBlock(d6[0], e1[0], 64, false);
    /// output
    auto g = (new UpSampling2D(2))->Call(d7);
    //g = (new ZeroPadding2D(2, 2, 2, 2))->Call(g);
    g = (new Conv2D(3, 3, 1, Tensor::GetPadding(Same, 3)))->Call(g);
    auto outImage = (new Activation(new Tanh()))->Call(g);

    auto model = new Flow({ inImage->Outputs()[0] }, outImage);
    return model;
}

//////////////////////////////////////////////////////////////////////////
ModelBase* Pix2Pix::CreatePatchDiscriminator(const Shape& imgShape, uint32_t patchSize, bool useMiniBatchDiscrimination /*= true*/)
{
    NEURO_ASSERT(imgShape.Width() == imgShape.Height(), "Input image must be square.");
    NEURO_ASSERT(imgShape.Width() % patchSize == 0, "Input image size is not divisible by patch size.");

    size_t nbPatches = (size_t)::pow(imgShape.Width() / patchSize, 2);
    uint32_t stride = 2;
    auto patchInput = new Input(Shape(patchSize, patchSize, imgShape.Depth()));

    uint32_t filtersStart = 64;
    size_t nbConv = int(floor(::log(patchSize) / ::log(2)));
    vector<uint32_t> filtersList(nbConv);
    for (int i = 0; i < nbConv; ++i)
        filtersList[i] = filtersStart * min<uint32_t>(8, (uint32_t)::pow(2, i));

    auto discOut = (new Conv2D(filtersList[0], 3, stride, Tensor::GetPadding(Same, 3), new LeakyReLU(0.2f)))->Call(patchInput->Outputs());

    for (uint32_t i = 1; i < filtersList.size(); ++i)
    {
        uint32_t filters = filtersList[i];
        discOut = (new Conv2D(filters, 3, stride, Tensor::GetPadding(Same, 3)))->Call(discOut);
        discOut = (new BatchNormalization())->Call(discOut);
        discOut = (new Activation(new LeakyReLU(0.2f)))->Call(discOut);
    }

    auto xFlat = (new Flatten())->Call(discOut);
    auto x = (new Dense(2, new Softmax()))->Call(xFlat);

    // this is single patch processing model
    auto patchDisc = new Flow(patchInput->Outputs(), { x[0], xFlat[0] }, "patch_disc");

    // generate final model for processing all patches
    auto imgInput = new Input(imgShape);

    // generate patches
    vector<TensorLike*> patches;

    for (int y = 0; y < ::sqrt(nbPatches); ++y)
        for (int x = 0; x < ::sqrt(nbPatches); ++x)
        {
            static auto patchExtract = [=](const vector<TensorLike*>& inputNodes)->vector<TensorLike*> { return { sub_tensor2d(inputNodes[0], patchSize, patchSize, patchSize * x, patchSize * y) }; };

            auto patch = (new Lambda(patchExtract))->Call(imgInput->Outputs())[0];
            patches.push_back(patch);
        }

    vector<TensorLike*> xList;
    vector<TensorLike*> xFlatList;

    for (size_t i = 0; i < patches.size(); ++i)
    {
        auto output = patchDisc->Call(patches[i], "patch_disc_" + to_string(i));
        xList.push_back(output[0]);
        xFlatList.push_back(output[1]);
    }

    TensorLike* xMerged;

    if (xList.size() > 1)
        xMerged = (new Concatenate(WidthAxis))->Call(xList)[0];
    else
        xMerged = xList[0];

    if (useMiniBatchDiscrimination)
    {
        static auto minb_disc = [](const vector<TensorLike*>& inputNodes) -> vector<TensorLike*>
        {
            NameScope scope("mini_batch_discriminator");
            auto x = inputNodes[0]; // x will be of shape [5, 100, 1, ?], where '?' is batch size
            // for first argument of difference we need to convert x to [1, 5, 100, ?] with simple batch reshape
            auto d1 = batch_reshape(x, Shape(1, x->GetShape().Width(), x->GetShape().Height()));
            // for second argument of difference we need to convert x to [?, 5, 100, 1] with a transpose
            auto d2 = transpose(x, { _3Axis, _0Axis, _1Axis, _2Axis });
            // due to broadcasting, the end result of following difference will be of shape [?, 5, 100, ?]
            auto diffs = sub(d1, d2);
            // first summation will produce a tensor with shape [?, 1, 100, ?]
            auto abs_diffs = sum(abs(diffs), _1Axis);
            // second summation will produce a tensor with shape [1, 1, 100, ?] and lastly we need a simple flatten to get 100 to the beginning [100, 1, 1, ?]
            return { batch_flatten(sum(exp(negative(abs_diffs)), _0Axis)) };
        };

        TensorLike* xFlatMerged;

        if (xFlatList.size() > 1)
            xFlatMerged = (new Concatenate(WidthAxis))->Call(xFlatList)[0];
        else
            xFlatMerged = xFlatList[0];

        uint32_t numKernels = 100;
        uint32_t dimPerKernel = 5;

        auto x_mbd = (new Dense(numKernels * dimPerKernel))->UseBias(false)->Call(xFlatMerged)[0];
        x_mbd = (new Reshape(Shape(dimPerKernel, numKernels)))->Call(x_mbd)[0];
        x_mbd = (new Lambda(minb_disc))->Call(x_mbd)[0];
        xMerged = (new Concatenate(WidthAxis))->Call({ xMerged, x_mbd })[0];
    }

    auto xOut = (new Dense(2, new Softmax()))->Call(xMerged);

    auto model = new Flow(imgInput->Outputs(), xOut, "disc");
    model->Optimize(new Adam(0.0001f), new BinaryCrossEntropy(), {}, Loss | Accuracy);
    return model;
}
