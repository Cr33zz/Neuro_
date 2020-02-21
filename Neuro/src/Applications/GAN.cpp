#include "Applications/GAN.h"
#include "Tensors/Tensor.h"
#include "Layers/Conv2D.h"
#include "Layers/Activation.h"
#include "Layers/BatchNormalization.h"
#include "Layers/Dropout.h"
#include "Layers/UpSampling2D.h"
#include "Layers/Concatenate.h"
#include "Layers/Padding2D.h"
#include "Layers/Input.h"
#include "Models/Flow.h"
#include "Activations.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    ModelBase* GAN::CreateUNetGenerator(const Shape& imgShape, uint32_t filtersStart)
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
    ModelBase* GAN::CreatePatchDiscriminator(const Shape& imgShape, uint32_t filtersStart)
    {
        auto inSrcImage = (new Input(imgShape))->Outputs()[0];
        auto inTargetImage = (new Input(imgShape))->Outputs()[0];

        auto merged = (new Concatenate(DepthAxis))->Call({ inSrcImage, inTargetImage })[0];

        auto block = [](TensorLike* x, uint32_t nbFilters, const string& name, bool batchNorm = true)
        {
            x = (new ZeroPadding2D(2, 1, 2, 1))->Call(x)[0];
            x = (new Conv2D(nbFilters, 4, 2))->Call(x)[0];
            if (batchNorm)
            {
                x = (new BatchNormalization())->Call(x)[0];
                x = (new Activation(new LeakyReLU(0.2f)))->Call(x)[0];
            }
            return x;
        };

        uint32_t minSize = min(imgShape.Width(), imgShape.Height());
        size_t nbConv = int(floor(::log(minSize) / ::log(2)));
        vector<uint32_t> filtersList(nbConv);
        for (int i = 0; i < nbConv; ++i)
            filtersList[i] = filtersStart * min<uint32_t>(8, (uint32_t)::pow(2, i));

        vector<TensorLike*> blocks;
        for (uint32_t i = 0; i < filtersList.size(); ++i)
        {
            uint32_t nbFilters = filtersList[i];
            string name = "patch_block_" + to_string(i + 1);
            blocks.push_back(block(i == 0 ? merged : blocks.back(), nbFilters, name, i > 0));
        }

        // 64
        //auto d = (new ZeroPadding2D(2, 1, 2, 1))->Call(merged);
        //d = (new Conv2D(filtersStart, 4, 2, 0, new LeakyReLU(0.2f)))->Call(d);
        //// 128
        //d = (new ZeroPadding2D(2, 1, 2, 1))->Call(d);
        //d = (new Conv2D(filtersStart * 2, 4, 2))->Call(d);
        //d = (new BatchNormalization())->Call(d);
        //d = (new Activation(new LeakyReLU(0.2f)))->Call(d);
        //// 256
        //d = (new ZeroPadding2D(2, 1, 2, 1))->Call(d);
        //d = (new Conv2D(filtersStart * 4, 4, 2))->Call(d);
        //d = (new BatchNormalization())->Call(d);
        //d = (new Activation(new LeakyReLU(0.2f)))->Call(d);
        //// 512
        //d = (new ZeroPadding2D(1, 1, 1, 1))->Call(d);
        //d = (new Conv2D(filtersStart * 8, 4, 1))->Call(d);
        //d = (new BatchNormalization())->Call(d);
        //d = (new Activation(new LeakyReLU(0.2f)))->Call(d);
        /*d = (new ZeroPadding2D(2, 1, 2, 1))->Call(d);
        d = (new Conv2D(512, 4, 1))->Call(d);
        d = (new BatchNormalization())->Call(d);
        d = (new Activation(new LeakyReLU(0.2f)))->Call(d);*/

        // patch output
        auto d = (new ZeroPadding2D(1, 1, 1, 1))->Call(blocks.back());
        d = (new Conv2D(1, 4, 1))->Call(d);
        auto patchOut = (new Activation(new Sigmoid()))->Call(d);

        auto model = new Flow({ inSrcImage, inTargetImage }, patchOut);
        return model;
    }
}