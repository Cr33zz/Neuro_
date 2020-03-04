#include "NeuralStyleTransfer.h"

//////////////////////////////////////////////////////////////////////////
TensorLike* GramMatrix(TensorLike* features, uint32_t area, uint32_t depth, bool normalize, const string& name)
{
    NameScope scope(name + "_gram");
    assert(features->GetShape().Batch() == 1);

    auto reshapedFeatures = reshape(features, Shape(area, depth));
    auto gram = matmul(reshapedFeatures, false, name);

    if (normalize)
        gram = div(gram, (float)reshapedFeatures->GetShape().Length, name + "_norm");

    return gram;
    //return div(matmul(features, transpose(features)), (float)featureMapSize, "result");
    //return matmul(reshapedFeatures, transpose(reshapedFeatures));
}

//////////////////////////////////////////////////////////////////////////
TensorLike* StyleLoss(TensorLike* styleFeatures, TensorLike* stylizedFeatures, int index, int mode)
{
    assert(stylizedFeatures->GetShape().Batch() == 1);
    auto& shape = styleFeatures->GetShape();
    uint32_t M = shape.Width() * shape.Height();
    uint32_t N = shape.Depth();

    return StyleLossFromGram(
        GramMatrix(styleFeatures, M, N, mode == 1, "style" + to_string(index)),
        GramMatrix(stylizedFeatures, M, N, mode == 1, "stylized" + to_string(index)),
        M,
        N,
        index, 
        mode);
}

//////////////////////////////////////////////////////////////////////////
TensorLike* StyleLossFromGram(TensorLike* styleGram, TensorLike* stylizedGram, uint32_t area, uint32_t depth, int index, int mode)
{
    NameScope scope("style_loss_" + to_string(index));
    
    auto loss = sum(square(sub(styleGram, stylizedGram)));

    if (mode > 1)
        loss = multiply(loss, 1.f / (4.f * ::pow((float)area, 2) * ::pow((float)depth, 2)));

    return loss;
}

//////////////////////////////////////////////////////////////////////////
TensorLike* ContentLoss(TensorLike* contentFeatures, TensorLike* stylizedFeatures, int mode)
{
    NameScope scope("content_loss");
    auto& shape = stylizedFeatures->GetShape();
    float M = (float)shape.Height() * shape.Width();
    float N = (float)shape.Depth();

    float K = 1.f;
    if (mode == 1)
        K = 1.f / (2.f * ::sqrt(N) * ::sqrt(M));
    else if (mode == 2)
        K = 1.f / (N * M);
    else if (mode == 3)
        K = 0.5f;

    return multiply(sum(square(sub(contentFeatures, stylizedFeatures))), K);
}
