#pragma once

#include "Layers/LayerBase.h"

namespace Neuro
{
    class UpSampling : public LayerBase
    {
    public:
        UpSampling(LayerBase* inputLayer, int filterSize, int stride = 1, const string& name = "");
        // Use this constructor for input layer only!
        UpSampling(Shape inputShape, int filterSize, int stride = 1, const string& name = "");

    protected:
        UpSampling();

        virtual LayerBase* GetCloneInstance() const override;
        virtual void OnClone(const LayerBase& source) override;
        virtual void FeedForwardInternal(bool training) override;
        virtual void BackPropInternal(Tensor& outputGradient) override;

    private:
        static Shape GetOutShape(const Shape& inputShape, int filterWidth, int filterHeight, int stride);

        int FilterSize;
        int Stride;
    };
}
