#include "NeuralStyleTransfer.h"

//////////////////////////////////////////////////////////////////////////
TensorLike* GramMatrix(TensorLike* features, const string& name)
{
    NameScope scope(name + "_gram");
    assert(features->GetShape().Batch() == 1);

    uint32_t featureMapSize = features->GetShape().Width() * features->GetShape().Height();
    auto reshapedFeatures = reshape(features, Shape(featureMapSize, features->GetShape().Depth()));
    return matmul(reshapedFeatures, transpose(reshapedFeatures), "result");
    //return div(matmul(reshapedFeatures, transpose(reshapedFeatures)), (float)reshapedFeatures->GetShape().Length, "result");
    //return div(matmul(features, transpose(features)), (float)featureMapSize, "result");
}

//////////////////////////////////////////////////////////////////////////
TensorLike* StyleLoss(TensorLike* styleGram, TensorLike* stylizedFeatures, int index)
{
    NameScope scope("style_loss_" + to_string(index));
    assert(stylizedFeatures->GetShape().Batch() == 1);

    auto stylizedGram = GramMatrix(stylizedFeatures, "gen_style_" + to_string(index));
    auto& styleFeaturesShape = stylizedFeatures->GetShape();
    float height = (float)styleFeaturesShape.Height(), width = (float)styleFeaturesShape.Width(), channels = (float)styleFeaturesShape.Depth();
    //return sum(square(sub(styleGram, stylizedGram)));
    return div(sum(square(sub(styleGram, stylizedGram))), 4.f * ::pow(channels, 2) * ::pow(width * height, 2));
}

//////////////////////////////////////////////////////////////////////////
TensorLike* ContentLoss(TensorLike* contentFeatures, TensorLike* stylizedFeatures)
{
    NameScope scope("content_loss");
    auto& shape = stylizedFeatures->GetShape();
    auto& contentFeaturesShape = contentFeatures->GetShape();
    float height = (float)contentFeaturesShape.Height(), width = (float)contentFeaturesShape.Width(), channels = (float)contentFeaturesShape.Depth();
    //return mean(square(sub(targetContentFeatures, contentFeatures)));
    //return l2_loss(sub(targetContentFeatures, contentFeatures));
    //return div(sum(square(sub(contentFeatures, stylizedFeatures))), (float)shape.Width() * shape.Height() * shape.Depth());
    return div(sum(square(sub(contentFeatures, stylizedFeatures))), 2.f * ::sqrt(channels) * ::sqrt(width * height));
}
