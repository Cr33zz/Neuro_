#include "NeuralStyleTransfer.h"

//////////////////////////////////////////////////////////////////////////
Neuro::TensorLike* NeuralStyleTransfer::GramMatrix(TensorLike* x, const string& name)
{
    NameScope scope(name + "_gram");
    assert(x->GetShape().Batch() == 1);

    uint32_t featureMapSize = x->GetShape().Width() * x->GetShape().Height();
    auto features = reshape(x, Shape(featureMapSize, x->GetShape().Depth()));
    return div(matmul(features, transpose(features)), (float)featureMapSize, "result");
}

//////////////////////////////////////////////////////////////////////////
Neuro::TensorLike* NeuralStyleTransfer::StyleLoss(TensorLike* targetStyleGram, TensorLike* styleFeatures, int index)
{
    NameScope scope("style_loss_" + to_string(index));
    assert(styleFeatures->GetShape().Batch() == 1);

    float channels = styleFeatures->GetShape().Depth();

    auto styleGram = GramMatrix(styleFeatures, "gen_style_" + to_string(index));
    return div(sum(l2_loss(sub(targetStyleGram, styleGram))), channels * channels);
}

//////////////////////////////////////////////////////////////////////////
Neuro::TensorLike* NeuralStyleTransfer::ContentLoss(TensorLike* targetContentFeatures, TensorLike* contentFeatures)
{
    NameScope scope("content_loss");
    return mean(square(sub(targetContentFeatures, contentFeatures)), GlobalAxis, "total");
}
