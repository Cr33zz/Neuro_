#include "Layers/Reshape.h"
#include "ComputationalGraph/Ops.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    Reshape::Reshape(const Shape& shape, const string& name)
        : Reshape(__FUNCTION__, shape, name)
    {
    }

    //////////////////////////////////////////////////////////////////////////
    Reshape::Reshape(const Shape& inputShape, const Shape& shape, const string& name)
        : Reshape(__FUNCTION__, inputShape, shape, name)
    {
    }

    //////////////////////////////////////////////////////////////////////////
    Reshape::Reshape(const string& constructorName, const Shape& shape, const string& name)
        : SingleLayer(constructorName, shape, nullptr, name), m_Shape(shape)
    {
        NEURO_ASSERT(shape.Batch() == 1, "");
    }

    //////////////////////////////////////////////////////////////////////////
    Reshape::Reshape(const string& constructorName, const Shape& inputShape, const Shape& shape, const string& name)
        : SingleLayer(constructorName, inputShape, nullptr, name), m_Shape(shape)
    {
        NEURO_ASSERT(shape.Batch() == 1, "");
    }

    //////////////////////////////////////////////////////////////////////////
    LayerBase* Reshape::GetCloneInstance() const
    {
        return new Reshape();
    }

    //////////////////////////////////////////////////////////////////////////
    vector<TensorLike*> Reshape::InternalCall(const vector<TensorLike*>& inputs, TensorLike* training)
    {
        return { batch_reshape(inputs[0], m_Shape) };
    }
}
