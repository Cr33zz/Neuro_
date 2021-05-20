#pragma once

#include "Layers/SingleLayer.h"

namespace Neuro
{
    class NEURO_DLL_EXPORT Concatenate : public SingleLayer
    {
    public:
        // Make sure to link this layer to input when using this constructor.
        Concatenate(EAxis axis, const string& name = "");

    protected:
        Concatenate() {}

        virtual LayerBase* GetCloneInstance() const override;
        
        virtual vector<TensorLike*> InternalCall(const vector<TensorLike*>& inputs) override;

    private:
        EAxis m_Axis;
    };
}
