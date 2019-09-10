#pragma once

#include "Layers/LayerBase.h"

namespace Neuro
{
    class Dropout : public LayerBase
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
        virtual void OnLink() override;
        virtual void FeedForwardInternal(bool training) override;
        virtual void BackPropInternal(vector<Tensor>& outputsGradient) override;

    private:
        Tensor m_Mask;
        float m_Prob;
    };
}
