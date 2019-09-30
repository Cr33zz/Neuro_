#pragma once

#include "Layers/SingleLayer.h"

namespace Neuro
{
    // https://www.youtube.com/watch?v=8oOgPUO-TBY
    class Pooling2D : public SingleLayer
    {
    public:
        Pooling2D(LayerBase* inputLayer, uint32_t filterSize, uint32_t stride = 1, uint32_t padding = 0, EPoolingMode mode = EPoolingMode::Max, EDataFormat dataFormat = NCHW, const string& name = "");
        // Make sure to link this layer to input when using this constructor.
        Pooling2D(uint32_t filterSize, uint32_t stride = 1, uint32_t padding = 0, EPoolingMode mode = EPoolingMode::Max, EDataFormat dataFormat = NCHW, const string& name = "");
        // Use this constructor for input layer only!
        Pooling2D(Shape inputShape, uint32_t filterSize, uint32_t stride = 1, uint32_t padding = 0, EPoolingMode mode = EPoolingMode::Max, EDataFormat dataFormat = NCHW, const string& name = "");

    protected:
        Pooling2D(const string& constructorName, LayerBase* inputLayer, uint32_t filterSize, uint32_t stride, uint32_t padding, EPoolingMode mode, EDataFormat dataFormat, const string& name);
        Pooling2D(const string& constructorName, Shape inputShape, uint32_t filterSize, uint32_t stride, uint32_t padding, EPoolingMode mode, EDataFormat dataFormat, const string& name);
        Pooling2D(const string& constructorName, uint32_t filterSize, uint32_t stride, uint32_t padding, EPoolingMode mode, EDataFormat dataFormat, const string& name);
        Pooling2D() {}

        virtual LayerBase* GetCloneInstance() const override;
        virtual void OnClone(const LayerBase& source) override;
        virtual void OnLinkInput(const vector<LayerBase*>& inputLayers) override;
        virtual void FeedForwardInternal(bool training) override;
        virtual void BackPropInternal(const tensor_ptr_vec_t& outputsGradient) override;

    private:
        EPoolingMode m_Mode;
        int m_FilterSize;
        int m_Stride;
        int m_Padding;
        EDataFormat m_DataFormat;
    };

    class MaxPooling2D : public Pooling2D
    {
    public:
        MaxPooling2D(LayerBase* inputLayer, uint32_t filterSize, uint32_t stride = 1, uint32_t padding = 0, EDataFormat dataFormat = NCHW, const string& name = "");
        // Make sure to link this layer to input when using this constructor.
        MaxPooling2D(uint32_t filterSize, uint32_t stride = 1, uint32_t padding = 0, EDataFormat dataFormat = NCHW, const string& name = "");
        // Use this constructor for input layer only!
        MaxPooling2D(Shape inputShape, uint32_t filterSize, uint32_t stride = 1, uint32_t padding = 0, EDataFormat dataFormat = NCHW, const string& name = "");
    };

    class AvgPooling2D : public Pooling2D
    {
    public:
        AvgPooling2D(LayerBase* inputLayer, uint32_t filterSize, uint32_t stride = 1, uint32_t padding = 0, EDataFormat dataFormat = NCHW, const string& name = "");
        // Make sure to link this layer to input when using this constructor.
        AvgPooling2D(uint32_t filterSize, uint32_t stride = 1, uint32_t padding = 0, EDataFormat dataFormat = NCHW, const string& name = "");
        // Use this constructor for input layer only!
        AvgPooling2D(Shape inputShape, uint32_t filterSize, uint32_t stride = 1, uint32_t padding = 0, EDataFormat dataFormat = NCHW, const string& name = "");
    };
}
