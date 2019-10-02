#pragma once

#include "Layers/SingleLayer.h"

namespace Neuro
{
    class Merge : public SingleLayer
    {
    public:
        Merge(const vector<LayerBase*>& inputLayers, EMergeMode mergeMode, ActivationBase* activation = nullptr, const string& name = "");
        // Make sure to link this layer to input when using this constructor.
        Merge(EMergeMode mergeMode, ActivationBase* activation = nullptr, const string& name = "");
        // This constructor should only be used for input layer
        Merge(const Shape& inputsShape, EMergeMode mergeMode, ActivationBase* activation = nullptr, const string& name = "");

    protected:
        Merge() {}

        virtual LayerBase* GetCloneInstance() const override;
        virtual void OnClone(const LayerBase& source) override;
        virtual void OnLinkInput(const vector<LayerBase*>& inputLayers) override;

        virtual void InitOps(TensorLike* training, bool initValues = true);

    private:
        EMergeMode m_MergeMode;
    };
}
