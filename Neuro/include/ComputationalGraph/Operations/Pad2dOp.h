#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    class Pad2dOp : public Operation
    {
    public:
        Pad2dOp(TensorLike* x, uint32_t left, uint32_t right, uint32_t top, uint32_t bottom, const string& name = "");

    protected:
        virtual void UpdateOutputShape() override;
        virtual void ComputeGradientInternal(const Tensor& grad) override;

        uint32_t m_Left;
        uint32_t m_Right;
        uint32_t m_Top;
        uint32_t m_Bottom;
    };

    class ConstantPad2dOp : public Pad2dOp
    {
    public:
        ConstantPad2dOp(TensorLike* x, uint32_t left, uint32_t right, uint32_t top, uint32_t bottom, float value, const string& name = "");

    protected:
        virtual void ComputeInternal() override;

    private:
        float m_Value;
    };

    class ReflectPad2dOp : public Pad2dOp
    {
    public:
        ReflectPad2dOp(TensorLike* x, uint32_t left, uint32_t right, uint32_t top, uint32_t bottom, const string& name = "");

    protected:
        virtual void ComputeInternal() override;
    };

    static Operation* constant_pad2d(TensorLike* x, uint32_t left, uint32_t right, uint32_t top, uint32_t bottom, float value, const string& name = "")
    {
        return new ConstantPad2dOp(x, left, right, top, bottom, value, name);
    }

    static Operation* reflect_pad2d(TensorLike* x, uint32_t left, uint32_t right, uint32_t top, uint32_t bottom, const string& name = "")
    {
        return new ReflectPad2dOp(x, left, right, top, bottom, name);
    }
}
