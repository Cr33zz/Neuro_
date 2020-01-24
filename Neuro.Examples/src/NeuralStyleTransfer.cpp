#include "NeuralStyleTransfer.h"

//////////////////////////////////////////////////////////////////////////
TensorLike* GramMatrix(TensorLike* features, const string& name)
{
    NameScope scope(name + "_gram");
    assert(features->GetShape().Batch() == 1);

    uint32_t featureMapSize = features->GetShape().Width() * features->GetShape().Height();
    auto reshapedFeatures = reshape(features, Shape(featureMapSize, features->GetShape().Depth()));
    return div(matmul(reshapedFeatures, transpose(reshapedFeatures)), (float)reshapedFeatures->GetShape().Length, "result");
    //return div(matmul(features, transpose(features)), (float)featureMapSize, "result");
    //return matmul(reshapedFeatures, transpose(reshapedFeatures));
}

//////////////////////////////////////////////////////////////////////////
TensorLike* StyleLoss(TensorLike* styleGram, TensorLike* stylizedFeatures, int index)
{
    assert(stylizedFeatures->GetShape().Batch() == 1);
    return StyleLossFromGram(styleGram, GramMatrix(stylizedFeatures, "stylized" + to_string(index)), stylizedFeatures->GetShape(), index);
}

//////////////////////////////////////////////////////////////////////////
TensorLike* StyleLossFromGram(TensorLike* styleGram, TensorLike* stylizedGram, const Shape& featuresShape, int index)
{
    NameScope scope("style_loss_" + to_string(index));
    
    return sum(square(sub(styleGram, stylizedGram)));
    //return mean(square(sub(styleGram, stylizedGram)));

    //float height = (float)featuresShape.Height(), width = (float)featuresShape.Width(), channels = (float)featuresShape.Depth();
    //return div(sum(square(sub(targetStyleGram, styleGram))), 4.f * ::pow(channels, 2) * ::pow(width * height, 2));
    //return div(sum(square(sub(styleGram, stylizedGram))), channels * width * height);
}

//////////////////////////////////////////////////////////////////////////
TensorLike* ContentLoss(TensorLike* contentFeatures, TensorLike* stylizedFeatures)
{
    NameScope scope("content_loss");
    auto& shape = stylizedFeatures->GetShape();
    //return mean(square(sub(contentFeatures, stylizedFeatures)));
    //return l2_loss(sub(contentFeatures, stylizedFeatures));
    return div(sum(square(sub(contentFeatures, stylizedFeatures))), (float)shape.Width() * shape.Height() * shape.Depth());
}
