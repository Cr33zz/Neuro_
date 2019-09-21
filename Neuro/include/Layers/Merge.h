#pragma once

#include "Layers/SingleLayer.h"

namespace Neuro
{
    class Merge : public SingleLayer
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
        virtual void OnLink(LayerBase* layers, bool input) override;
        virtual void FeedForwardInternal(bool training) override;
        virtual void BackPropInternal(const tensor_ptr_vec_t& outputsGradient) override;

    private:
        Mode m_MergeMode;
    };
}
