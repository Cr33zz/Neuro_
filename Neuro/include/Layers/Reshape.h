#pragma once

#include "Layers/SingleLayer.h"

namespace Neuro
{
    class NEURO_DLL_EXPORT Reshape : public SingleLayer
    {
    public:
        Reshape(const Shape& shape, const string& name = "");
        Reshape(const Shape& inputShape, const Shape& shape, const string& name = "");

    protected:
        Reshape(const string& constructorName, const Shape& shape, const string& name);
        Reshape(const string& constructorName, const Shape& inputShape, const Shape& shape, const string& name);
        Reshape() {}

        virtual LayerBase* GetCloneInstance() const override;

        virtual vector<TensorLike*> InternalCall(const vector<TensorLike*>& inputs) override;

        Shape m_Shape;
    };
}
