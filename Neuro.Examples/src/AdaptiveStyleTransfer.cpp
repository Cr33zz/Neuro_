#include "AdaptiveStyleTransfer.h"

//////////////////////////////////////////////////////////////////////////
ModelBase* AdaptiveStyleTransfer::CreateGeneratorModel(TensorLike* contentPre, TensorLike* styleContentFeat, TensorLike* alpha, Flow& vggEncoder)
{
    NameScope scope("generator");

    auto inputContent = new Input(contentPre, "input_content");
    auto inputStyle = new Input(styleContentFeat, "input_style");

    // encoder

    auto contentFeat = vggEncoder(inputContent->Outputs(), "content_features").back();

    // adaptive instance normalization

    auto adaInFeat = (new AdaIN(alpha))->Call({ contentFeat, inputStyle->Outputs()[0] }, "ada_in")[0];

    // decoder

    auto d = (new Conv2D(256, 3, 1, Tensor::GetPadding(Same, 3), new ReLU(), NCHW, "decode_block4_conv1"))->Call(adaInFeat)[0];
    d = (new UpSampling2D(2, "decode_up_1"))->Call(d)[0];
    
    d = (new Conv2D(256, 3, 1, Tensor::GetPadding(Same, 3), new ReLU(), NCHW, "decode_block3_conv4"))->Call(d)[0];
    d = (new Conv2D(256, 3, 1, Tensor::GetPadding(Same, 3), new ReLU(), NCHW, "decode_block3_conv3"))->Call(d)[0];
    d = (new Conv2D(256, 3, 1, Tensor::GetPadding(Same, 3), new ReLU(), NCHW, "decode_block3_conv2"))->Call(d)[0];
    d = (new Conv2D(128, 3, 1, Tensor::GetPadding(Same, 3), new ReLU(), NCHW, "decode_block3_conv1"))->Call(d)[0];
    d = (new UpSampling2D(2, "decode_up_2"))->Call(d)[0];

    d = (new Conv2D(128, 3, 1, Tensor::GetPadding(Same, 3), new ReLU(), NCHW, "decode_block2_conv2"))->Call(d)[0];
    d = (new Conv2D(64, 3, 1, Tensor::GetPadding(Same, 3), new ReLU(), NCHW, "decode_block2_conv1"))->Call(d)[0];
    d = (new UpSampling2D(2, "decode_up_3"))->Call(d)[0];

    d = (new Conv2D(64, 3, 1, Tensor::GetPadding(Same, 3), new ReLU(), NCHW, "decode_block1_conv2"))->Call(d)[0];
    d = (new Conv2D(3, 3, 1, Tensor::GetPadding(Same, 3), nullptr, NCHW, "decode_block1_conv1"))->Call(d)[0];

    return new Flow({ inputContent->Outputs()[0], inputStyle->Outputs()[0] }, { d, adaInFeat }, "generator_model");
}