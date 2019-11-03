#include "NeuralStyleTransfer.h"

//////////////////////////////////////////////////////////////////////////
Neuro::TensorLike* NeuralStyleTransfer::GramMatrix(TensorLike* x, const string& name)
{
    NameScope scope(name + "_gram");
    assert(x->GetShape().Batch() == 1);

    uint32_t featureMapSize = x->GetShape().Width() * x->GetShape().Height();
    auto features = reshape(x, Shape(featureMapSize, x->GetShape().Depth()));
    return div(matmul(features, transpose(features)), (float)features->GetShape().Length, "result");
    //return div(matmul(features, transpose(features)), (float)featureMapSize, "result");
}

//////////////////////////////////////////////////////////////////////////
Neuro::TensorLike* NeuralStyleTransfer::StyleLoss(TensorLike* targetStyleGram, TensorLike* styleFeatures, int index)
{
    NameScope scope("style_loss_" + to_string(index));
    assert(styleFeatures->GetShape().Batch() == 1);

    auto styleGram = GramMatrix(styleFeatures, "gen_style_" + to_string(index));
    auto& styleFeaturesShape = styleFeatures->GetShape();
    float height = (float)styleFeaturesShape.Height(), width = (float)styleFeaturesShape.Width(), channels = (float)styleFeaturesShape.Depth();
    return sum(square(sub(targetStyleGram, styleGram)));
    //return mean(square(sub(targetStyleGram, styleGram)));
    //return div(sum(square(sub(targetStyleGram, styleGram))), 4.f * ::pow(channels, 2) * ::pow(width * height, 2));
}

//////////////////////////////////////////////////////////////////////////
Neuro::TensorLike* NeuralStyleTransfer::ContentLoss(TensorLike* targetContentFeatures, TensorLike* contentFeatures)
{
    NameScope scope("content_loss");
    auto& shape = contentFeatures->GetShape();
    //return mean(square(sub(targetContentFeatures, contentFeatures)));
    //return l2_loss(sub(targetContentFeatures, contentFeatures));
    return div(sum(square(sub(targetContentFeatures, contentFeatures))), (float)shape.Width() * shape.Height() * shape.Depth());
}
