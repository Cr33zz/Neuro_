#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    class Pad2dOp : public Operation
    {
    public:
        Pad2dOp(TensorLike* x, uint32_t left, uint32_t right, uint32_t top, uint32_t bottom, float value, const string& name = "");

    protected:
        virtual void ComputeInternal() override;
        virtual void ComputeGradientInternal(const Tensor& grad) override;

    private:
        uint32_t m_Left;
        uint32_t m_Right;
        uint32_t m_Top;
        uint32_t m_Bottom;
        float m_Value;
    };

    static Operation* pad2d(TensorLike* x, uint32_t left, uint32_t right, uint32_t top, uint32_t bottom, float value, const string& name = "")
    {
        return new Pad2dOp(x, left, right, top, bottom, value, name);
    }
}
