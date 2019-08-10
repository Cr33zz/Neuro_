#pragma once

#include "Layers/LayerBase.h"

namespace Neuro
{
    // https://www.youtube.com/watch?v=8oOgPUO-TBY
    class Pooling : public LayerBase
    {
    public:
        Pooling(LayerBase* inputLayer, int filterSize, int stride = 1, Tensor::EPoolType type = Tensor::EPoolType::Max, const string& name = "");
        // Use this constructor for input layer only!
        Pooling(Shape inputShape, int filterSize, int stride = 1, Tensor::EPoolType type = Tensor::EPoolType::Max, const string& name = "");

        virtual const char* ClassName() const override;

    protected:
        Pooling();

        virtual LayerBase* GetCloneInstance() const override;
        virtual void OnClone(const LayerBase& source) override;
        virtual void FeedForwardInternal() override;
        virtual void BackPropInternal(Tensor& outputGradient) override;

    private:
        static Shape GetOutShape(const Shape& inputShape, int filterWidth, int filterHeight, int stride);

        Tensor::EPoolType Type;
        int FilterSize;
        int Stride;
    };
}
