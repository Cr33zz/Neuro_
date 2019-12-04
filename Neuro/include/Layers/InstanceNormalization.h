#pragma once

#include "Layers/BatchNormalization.h"

namespace Neuro
{
    class InstanceNormalization : public BatchNormalization
    {
    public:
        // Make sure to link this layer to input when using this constructor.
        InstanceNormalization(const string& name = "");
        // This constructor should only be used for input layer
        InstanceNormalization(const Shape& inputShape, const string& name = "");

    protected:
        virtual void Build(const vector<Shape>& inputShapes) override;
        virtual vector<TensorLike*> InternalCall(const vector<TensorLike*>& inputs) override;
    };
}
