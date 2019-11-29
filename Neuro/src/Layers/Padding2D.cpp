#include "Layers/Padding2D.h"
#include "ComputationalGraph/Ops.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    Padding2D::Padding2D(uint32_t left, uint32_t right, uint32_t top, uint32_t bottom, float value, const string& name)
        : SingleLayer(__FUNCTION__, Shape(), nullptr, name), m_Left(left), m_Right(right), m_Top(top), m_Bottom(bottom), m_Value(value)
    {
    }

    //////////////////////////////////////////////////////////////////////////
    vector<TensorLike*> Padding2D::InternalCall(const vector<TensorLike*>& inputs, TensorLike* training)
    {
        return { constant_pad2d(inputs[0], m_Left, m_Right, m_Top, m_Bottom, m_Value) };
    }
}
