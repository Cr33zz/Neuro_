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
        virtual void OnLink(LayerBase* layer, bool input) override;
        virtual void FeedForwardInternal(bool training) override;
        virtual void BackPropInternal(vector<Tensor>& outputsGradient) override;

    private:
        Tensor m_Mask;
        float m_Prob;
    };
}
