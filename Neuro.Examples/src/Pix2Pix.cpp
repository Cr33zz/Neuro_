#include "Pix2Pix.h"

//////////////////////////////////////////////////////////////////////////
ModelBase* Pix2Pix::CreateGenerator(const Shape& imgShape)
{
    auto encoderBlock = [](TensorLike* x, uint32_t nbFilters, const string& name, bool batchNorm = true)
    {
        x = (new Activation(new LeakyReLU(0.2f)))->Call(x)[0];
        x = (new Conv2D(nbFilters, 3, 2, Tensor::GetPadding(Same, 3), nullptr, NCHW, name))->Call(x)[0];
        if (batchNorm)
            x = (new BatchNormalization())->Call(x)[0];
        return x;
    };

    auto decoderBlock = [](TensorLike* x, TensorLike* x2, uint32_t nbFilters, const string& name, bool batchNorm = true, bool dropout = false)
    {
        x = (new Activation(new ReLU()))->Call(x)[0];
        x = (new UpSampling2D(2))->Call(x)[0];
        x = (new Conv2D(nbFilters, 3, 1, Tensor::GetPadding(Same, 3), nullptr, NCHW, name))->Call(x)[0];
        if (batchNorm)
            x = (new BatchNormalization())->Call(x)[0];
        if (dropout)
            x = (new Dropout(0.5f))->Call(x)[0];
        x = (new Concatenate(DepthAxis))->Call({ x, x2 })[0];
        return x;
    };

    uint32_t minSize = min(imgShape.Width(), imgShape.Height());
    uint32_t filtersStart = 64;
    size_t nbConv = int(floor(::log(minSize) / ::log(2)));
    vector<uint32_t> filtersList(nbConv);
    for (int i = 0; i < nbConv; ++i)
        filtersList[i] = filtersStart * min<uint32_t>(8, (uint32_t)::pow(2, i));

    auto inImage = new Input(imgShape);

    // Encoder
    vector<TensorLike*> encoderList = { (new Conv2D(filtersList[0], 3, 2, Tensor::GetPadding(Same, 3), nullptr, NCHW, "unet_conv2D_1"))->Call(inImage->Outputs())[0] };
    for (uint32_t i = 1; i < filtersList.size(); ++i)
    {
        uint32_t nbFilters = filtersList[i];
        string name = "unet_conv2D_" + to_string(i + 1);
        encoderList.push_back(encoderBlock(encoderList.back(), nbFilters, name));
    }

    // Prepare decoder filters
    filtersList.pop_back();
    filtersList.pop_back();
    reverse(filtersList.begin(), filtersList.end());
    if (filtersList.size() < nbConv - 1)
        filtersList.push_back(filtersStart);

    // Decoder
    vector<TensorLike*> decoderList = { decoderBlock(encoderList.back(), *(encoderList.end() - 2), filtersList[0], "unet_upconv2D_1", true, true) };
    for (uint32_t i = 1; i < filtersList.size(); ++i)
    {
        uint32_t nbFilters = filtersList[i];
        string name = "unet_upconv2D_" + to_string(i + 1);
        // Dropout only on first few layers
        bool d = i < 3;
        decoderList.push_back(decoderBlock(decoderList.back(), *(encoderList.end() - (i + 2)), nbFilters, name, true, d));
    }
    
    auto x = (new Activation(new ReLU()))->Call(decoderList.back())[0];
    x = (new UpSampling2D(2))->Call(x)[0];
    x = (new Conv2D(imgShape.Depth(), 3, 1, Tensor::GetPadding(Same, 3), nullptr, NCHW, "last_conv"))->Call(x)[0];
    x = (new Activation(new Tanh()))->Call(x)[0];

    auto model = new Flow(inImage->Outputs(), { x }, "gen");
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
    return model;
}
