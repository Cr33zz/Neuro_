#pragma once

#include "Layers/LayerBase.h"

namespace Neuro
{
    class Concatenate : public LayerBase
    {
    public:
        Concatenate(const vector<LayerBase*>& inputLayers, const string& name = "");

        const char* ClassName() const;

    protected:
        Concatenate();

        virtual LayerBase* GetCloneInstance() const override;
        virtual void FeedForwardInternal() override;
        virtual void BackPropInternal(Tensor& outputGradient) override;
    };
}
