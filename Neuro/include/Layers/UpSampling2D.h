#pragma once

#include "Layers/SingleLayer.h"

namespace Neuro
{
    class UpSampling2D : public SingleLayer
    {
    public:
        UpSampling2D(LayerBase* inputLayer, uint32_t scaleFactor, const string& name = "");
        // Make sure to link this layer to input when using this constructor.
        UpSampling2D(uint32_t scaleFactor, const string& name = "");
        // Use this constructor for input layer only!
        UpSampling2D(const Shape& inputShape, uint32_t scaleFactor, const string& name = "");

    protected:
        UpSampling2D() {}

        virtual LayerBase* GetCloneInstance() const override;
        virtual void OnClone(const LayerBase& source) override;
        virtual void OnLinkInput(const vector<LayerBase*>& inputLayers) override;

        virtual void InitOps(TensorLike* training, bool initValues = true);

    private:
        int m_ScaleFactor;
    };
}
