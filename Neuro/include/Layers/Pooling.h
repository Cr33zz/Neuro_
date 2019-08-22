#pragma once

#include "Layers/LayerBase.h"

namespace Neuro
{
    // https://www.youtube.com/watch?v=8oOgPUO-TBY
    class Pooling : public LayerBase
    {
    public:
        Pooling(LayerBase* inputLayer, int filterSize, int stride = 1, EPoolingMode type = EPoolingMode::Max, const string& name = "");
        // Use this constructor for input layer only!
        Pooling(Shape inputShape, int filterSize, int stride = 1, EPoolingMode type = EPoolingMode::Max, const string& name = "");

    protected:
        Pooling();

        virtual LayerBase* GetCloneInstance() const override;
        virtual void OnClone(const LayerBase& source) override;
        virtual void FeedForwardInternal(bool training) override;
        virtual void BackPropInternal(Tensor& outputGradient) override;

    private:
        static Shape GetOutShape(const Shape& inputShape, int filterWidth, int filterHeight, int stride);

        EPoolingMode Type;
        int FilterSize;
        int Stride;
    };

    class MaxPooling : public Pooling
    {
    public:
        MaxPooling(LayerBase* inputLayer, int filterSize, int stride = 1, const string& name = "");
        // Use this constructor for input layer only!
        MaxPooling(Shape inputShape, int filterSize, int stride = 1, const string& name = "");
    };

    class AvgPooling : public Pooling
    {
    public:
        AvgPooling(LayerBase* inputLayer, int filterSize, int stride = 1, const string& name = "");
        // Use this constructor for input layer only!
        AvgPooling(Shape inputShape, int filterSize, int stride = 1, const string& name = "");
    };
}
