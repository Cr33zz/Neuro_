#pragma once

#include "Layers/SingleLayer.h"

namespace Neuro
{
    class Dropout : public SingleLayer
    {
    public:
        Dropout(float p, const string& name = "");
        // This constructor should only be used for input layer
        Dropout(const Shape& inputShape, float p, const string& name = "");

    protected:
        Dropout() {}

        virtual LayerBase* GetCloneInstance() const override;
        
        virtual vector<TensorLike*> InternalCall(const vector<TensorLike*>& inputs) override;

    private:
        float m_Prob;
    };
}
