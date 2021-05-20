#pragma once

#include "Layers/SingleLayer.h"

namespace Neuro
{
    class NEURO_DLL_EXPORT Merge : public SingleLayer
    {
    public:
        Merge(EMergeMode mergeMode, ActivationBase* activation = nullptr, const string& name = "");
        // This constructor should only be used for input layer
        Merge(const Shape& inputsShape, EMergeMode mergeMode, ActivationBase* activation = nullptr, const string& name = "");

    protected:
        Merge() {}

        virtual LayerBase* GetCloneInstance() const override;
        virtual void OnClone(const LayerBase& source) override;
        
        virtual vector<TensorLike*> InternalCall(const vector<TensorLike*>& inputs) override;

    private:
        EMergeMode m_MergeMode;
    };
}
