#include "NeuralStyleTransfer.h"

//////////////////////////////////////////////////////////////////////////
Neuro::TensorLike* NeuralStyleTransfer::GramMatrix(TensorLike* x, const string& name)
{
    NameScope scope(name + "_gram");
    assert(x->GetShape().Batch() == 1);

    uint32_t elementsPerFeature = x->GetShape().Width() * x->GetShape().Height();
    auto features = reshape(x, Shape(elementsPerFeature, x->GetShape().Depth()));
    return multiply(matmul(features, transpose(features)), 1.f / x->GetShape().Length, "result");
}

//////////////////////////////////////////////////////////////////////////
Neuro::TensorLike* NeuralStyleTransfer::StyleLoss(TensorLike* targetStyleGram, TensorLike* styleFeatures, int index)
{
    NameScope scope("style_loss_" + to_string(index));
    assert(styleFeatures->GetShape().Batch() == 1);

    auto styleGram = GramMatrix(styleFeatures, "gen_style_" + to_string(index));
    return mean(square(sub(targetStyleGram, styleGram)), GlobalAxis, "total");
}

//////////////////////////////////////////////////////////////////////////
Neuro::TensorLike* NeuralStyleTransfer::ContentLoss(TensorLike* targetContentFeatures, TensorLike* contentFeatures)
{
    NameScope scope("content_loss");
    return mean(square(sub(targetContentFeatures, contentFeatures)), GlobalAxis, "total");
}
