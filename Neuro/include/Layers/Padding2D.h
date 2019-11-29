#pragma once

#include "Layers/SingleLayer.h"

namespace Neuro
{
    class Padding2D : public SingleLayer
    {
    public:
        Padding2D(uint32_t left, uint32_t right, uint32_t top, uint32_t bottom, float value, const string& name = "");

    protected:
        Padding2D() {}

        virtual vector<TensorLike*> InternalCall(const vector<TensorLike*>& inputs, TensorLike* training) override;

    private:
        uint32_t m_Left;
        uint32_t m_Right;
        uint32_t m_Top;
        uint32_t m_Bottom;
        float m_Value;
    };
}
