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
        // This constructor should only be used for input layer
        //Merge(const vector<Shape>& inputShapes, Mode mergeMode, const string& name = "");

        virtual const char* ClassName() const override;

    protected:
        Merge();

        virtual LayerBase* GetCloneInstance() const override;
        virtual void OnClone(const LayerBase& source) override;
        virtual void FeedForwardInternal() override;
        virtual void BackPropInternal(Tensor& outputGradient) override;

    private:
        Mode MergeMode;
    };
}
