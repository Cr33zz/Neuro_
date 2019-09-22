#pragma once

#include "Layers/SingleLayer.h"

namespace Neuro
{
    class Dropout : public SingleLayer
    {
    public:
        Dropout(LayerBase* inputLayer, float p, const string& name = "");
        // Make sure to link this layer to input when using this constructor.
        Dropout(float p, const string& name = "");
        // This constructor should only be used for input layer
        Dropout(const Shape& inputShape, float p, const string& name = "");

    protected:
        Dropout() {}

        virtual LayerBase* GetCloneInstance() const override;
        virtual void OnLinkInput(const vector<LayerBase*>& inputLayers) override;
        virtual void FeedForwardInternal(bool training) override;
        virtual void BackPropInternal(const tensor_ptr_vec_t& outputsGradient) override;

    private:
        Tensor m_Mask;
        float m_Prob;
    };
}
