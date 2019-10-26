#include "NeuralStyleTransfer.h"

//////////////////////////////////////////////////////////////////////////
Neuro::TensorLike* NeuralStyleTransfer::GramMatrix(TensorLike* x, const string& name)
{
    NameScope scope(name + "_gram");
    assert(x->GetShape().Batch() == 1);

    uint32_t elementsPerFeature = x->GetShape().Width() * x->GetShape().Height();
    auto features = reshape(x, Shape(elementsPerFeature, x->GetShape().Depth()));
    return multiply(matmul(features, transpose(features)), 1.f / elementsPerFeature, "result");
}

//////////////////////////////////////////////////////////////////////////
Neuro::TensorLike* NeuralStyleTransfer::StyleLoss(TensorLike* styleGram, TensorLike* gen, int index)
{
    NameScope scope("style_loss_" + to_string(index));
    assert(gen->GetShape().Batch() == 1);

    //auto s = GramMatrix(style, index);
    auto genGram = GramMatrix(gen, "gen_style_" + to_string(index));

    float channels = (float)gen->GetShape().Depth();
    float size = (float)(gen->GetShape().Height() * gen->GetShape().Width());

    //return multiply(mean(square(sub(styleGram, genGram))), 1.f / (4.f * (channels * channels) * (size * size)), "total");
    return mean(square(sub(styleGram, genGram)), GlobalAxis, "total");
}

//////////////////////////////////////////////////////////////////////////
Neuro::TensorLike* NeuralStyleTransfer::ContentLoss(TensorLike* content, TensorLike* gen)
{
    NameScope scope("content_loss");
    return mean(square(sub(gen, content)), GlobalAxis, "total");
}
