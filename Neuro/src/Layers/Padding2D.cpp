#include "Layers/Padding2D.h"
#include "ComputationalGraph/Ops.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    Padding2D::Padding2D(uint32_t left, uint32_t right, uint32_t top, uint32_t bottom, const string& name)
        : Padding2D(__FUNCTION__, left, right, top, bottom, name)
    {
    }

    //////////////////////////////////////////////////////////////////////////
    Padding2D::Padding2D(const string& constructorName, uint32_t left, uint32_t right, uint32_t top, uint32_t bottom, const string& name)
        : SingleLayer(__FUNCTION__, Shape(), nullptr, name), m_Left(left), m_Right(right), m_Top(top), m_Bottom(bottom)
    {
    }

    //////////////////////////////////////////////////////////////////////////
    ConstantPadding2D::ConstantPadding2D(uint32_t left, uint32_t right, uint32_t top, uint32_t bottom, float value, const string& name)
        : ConstantPadding2D(__FUNCTION__, left, right, top, bottom, value, name)
    {
    }

    //////////////////////////////////////////////////////////////////////////
    ConstantPadding2D::ConstantPadding2D(const string& constructorName, uint32_t left, uint32_t right, uint32_t top, uint32_t bottom, float value, const string& name)
        : Padding2D(constructorName, left, right, top, bottom, name), m_Value(value)
    {
    }

    //////////////////////////////////////////////////////////////////////////
    vector<TensorLike*> ConstantPadding2D::InternalCall(const vector<TensorLike*>& inputs, TensorLike* training)
    {
        return { constant_pad2d(inputs[0], m_Left, m_Right, m_Top, m_Bottom, m_Value) };
    }

    //////////////////////////////////////////////////////////////////////////
    ZeroPadding2D::ZeroPadding2D(uint32_t left, uint32_t right, uint32_t top, uint32_t bottom, const string& name)
        : ConstantPadding2D(__FUNCTION__, left, right, top, bottom, 0.f, name)
    {
    }

    //////////////////////////////////////////////////////////////////////////
    ReflectPadding2D::ReflectPadding2D(uint32_t left, uint32_t right, uint32_t top, uint32_t bottom, const string& name)
        : Padding2D(__FUNCTION__, left, right, top, bottom, name)
    {
    }

    //////////////////////////////////////////////////////////////////////////
    vector<TensorLike*> ReflectPadding2D::InternalCall(const vector<TensorLike*>& inputs, TensorLike* training)
    {
        return { reflect_pad2d(inputs[0], m_Left, m_Right, m_Top, m_Bottom) };
    }
}
