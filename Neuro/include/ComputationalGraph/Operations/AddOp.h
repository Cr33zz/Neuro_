#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    class AddOp : public Operation
    {
    public:
        AddOp(TensorLike* a, TensorLike* b, const string& name = "");
        AddOp(TensorLike* x, float val, const string& name = "");

    protected:
        virtual void UpdateOutputShape() override;
        virtual void ComputeInternal() override;
        virtual void ComputeGradientInternal(const Tensor& grad) override;

    private:
        float m_Val = 0.f;
    };

    static Operation* add(TensorLike* a, TensorLike* b, const string& name = "")
    {
        return new AddOp(a, b, name);
    }

    static Operation* add(TensorLike* x, float val, const string& name = "")
    {
        return new AddOp(x, val, name);
    }
}
