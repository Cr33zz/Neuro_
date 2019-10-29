#include "NeuralStyleTransfer.h"

//////////////////////////////////////////////////////////////////////////
Neuro::TensorLike* NeuralStyleTransfer::GramMatrix(TensorLike* x, const string& name)
{
    NameScope scope(name + "_gram");
    assert(x->GetShape().Batch() == 1);

    uint32_t featureMapSize = x->GetShape().Width() * x->GetShape().Height();
    auto features = reshape(x, Shape(featureMapSize, x->GetShape().Depth()));
    return div(matmul(features, transpose(features)), (float)features->GetShape().Length, "result");
}

//////////////////////////////////////////////////////////////////////////
Neuro::TensorLike* NeuralStyleTransfer::StyleLoss(TensorLike* targetStyleGram, TensorLike* styleFeatures, int index)
{
    NameScope scope("style_loss_" + to_string(index));
    assert(styleFeatures->GetShape().Batch() == 1);

    float channels = (float)styleFeatures->GetShape().Depth();
    auto styleGram = GramMatrix(styleFeatures, "gen_style_" + to_string(index));
    return sum(square(sub(targetStyleGram, styleGram)));
}

//////////////////////////////////////////////////////////////////////////
Neuro::TensorLike* NeuralStyleTransfer::ContentLoss(TensorLike* targetContentFeatures, TensorLike* contentFeatures)
{
    NameScope scope("content_loss");
    auto& shape = contentFeatures->GetShape();
    return div(sum(square(sub(targetContentFeatures, contentFeatures))), shape.Width() * shape.Height() * shape.Depth());
}
