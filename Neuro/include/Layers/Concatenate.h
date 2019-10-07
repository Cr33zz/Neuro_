#pragma once

#include "Layers/SingleLayer.h"

namespace Neuro
{
    class Concatenate : public SingleLayer
    {
    public:
        Concatenate(const vector<LayerBase*>& inputLayers, EAxis axis, const string& name = "");
        // Make sure to link this layer to input when using this constructor.
        Concatenate(EAxis axis, const string& name = "");

    protected:
        Concatenate() {}

        virtual LayerBase* GetCloneInstance() const override;
        virtual void OnLinkInput(const vector<LayerBase*>& inputLayers) override;

        virtual void InternalCall(TensorLike* training, bool initValues = true) override;

    private:
        EAxis m_Axis;
    };
}
