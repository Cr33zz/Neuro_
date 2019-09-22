﻿#pragma once

#include "Layers/SingleLayer.h"

namespace Neuro
{
    class Merge : public SingleLayer
    {
    public:
        Merge(const vector<LayerBase*>& inputLayers, EMergeMode mergeMode, const string& name = "");
        // Make sure to link this layer to input when using this constructor.
        Merge(EMergeMode mergeMode, const string& name = "");
        // This constructor should only be used for input layer
        Merge(const Shape& inputsShape, EMergeMode mergeMode, const string& name = "");

    protected:
        Merge() {}

        virtual LayerBase* GetCloneInstance() const override;
        virtual void OnClone(const LayerBase& source) override;
        virtual void OnLink(LayerBase* layers, bool input) override;
        virtual void FeedForwardInternal(bool training) override;
        virtual void BackPropInternal(const tensor_ptr_vec_t& outputsGradient) override;

    private:
        EMergeMode m_MergeMode;
    };
}
