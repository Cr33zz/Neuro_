#pragma once

#include "Layers/LayerBase.h"

namespace Neuro
{
    class Merge : public LayerBase
    {
    public:
        enum Mode
        {
            Sum,
            Avg,
            Max,
            Min
        };

        Merge(const vector<LayerBase*>& inputLayers, Mode mergeMode, const string& name = "");
        // Make sure to link this layer to input when using this constructor.
        Merge(Mode mergeMode, const string& name = "");
        // This constructor should only be used for input layer
        Merge(const Shape& inputsShape, Mode mergeMode, const string& name = "");

    protected:
        Merge() {}

        virtual LayerBase* GetCloneInstance() const override;
        virtual void OnClone(const LayerBase& source) override;
        virtual void OnLink() override;
        virtual void FeedForwardInternal(bool training) override;
        virtual void BackPropInternal(vector<Tensor>& outputGradients) override;

    private:
        Mode m_MergeMode;
    };
}
