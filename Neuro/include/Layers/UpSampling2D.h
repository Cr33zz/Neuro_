#pragma once

#include "Layers/SingleLayer.h"

namespace Neuro
{
    class UpSampling2D : public SingleLayer
    {
    public:
        UpSampling2D(uint32_t scaleFactor, const string& name = "");
        // Use this constructor for input layer only!
        UpSampling2D(const Shape& inputShape, uint32_t scaleFactor, const string& name = "");

    protected:
        UpSampling2D() {}

        virtual LayerBase* GetCloneInstance() const override;
        virtual void OnClone(const LayerBase& source) override;
        
        virtual vector<TensorLike*> InternalCall(const vector<TensorLike*>& inputs, TensorLike* training) override;

    private:
        int m_ScaleFactor;
    };
}
