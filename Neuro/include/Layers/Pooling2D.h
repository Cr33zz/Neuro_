#pragma once

#include "Layers/LayerBase.h"

namespace Neuro
{
    // https://www.youtube.com/watch?v=8oOgPUO-TBY
    class Pooling2D : public LayerBase
    {
    public:
        Pooling2D(LayerBase* inputLayer, int filterSize, int stride = 1, int padding = 0, EPoolingMode mode = EPoolingMode::Max, const string& name = "");
        // Use this constructor for input layer only!
        Pooling2D(Shape inputShape, int filterSize, int stride = 1, int padding = 0, EPoolingMode mode = EPoolingMode::Max, const string& name = "");

    protected:
        Pooling2D();

        virtual LayerBase* GetCloneInstance() const override;
        virtual void OnClone(const LayerBase& source) override;
        virtual void FeedForwardInternal(bool training) override;
        virtual void BackPropInternal(Tensor& outputGradient) override;

    private:
        EPoolingMode m_Mode;
        int m_FilterSize;
        int m_Stride;
        int m_Padding;
    };

    class MaxPooling2D : public Pooling2D
    {
    public:
        MaxPooling2D(LayerBase* inputLayer, int filterSize, int stride = 1, int padding = 0, const string& name = "");
        // Use this constructor for input layer only!
        MaxPooling2D(Shape inputShape, int filterSize, int stride = 1, int padding = 0, const string& name = "");
    };

    class AvgPooling2D : public Pooling2D
    {
    public:
        AvgPooling2D(LayerBase* inputLayer, int filterSize, int stride = 1, int padding = 0, const string& name = "");
        // Use this constructor for input layer only!
        AvgPooling2D(Shape inputShape, int filterSize, int stride = 1, int padding = 0, const string& name = "");
    };
}
