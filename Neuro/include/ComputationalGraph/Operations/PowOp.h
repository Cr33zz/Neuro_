#pragma once

#include "ComputationalGraph/Operation.h"
#include "ComputationalGraph/Constant.h"

namespace Neuro
{
    class NEURO_DLL_EXPORT PowOp : public Operation
    {
    public:
        PowOp(TensorLike* x, TensorLike* p, const string& name = "");
        PowOp(TensorLike* x, float p, const string& name = "");

    protected:
        virtual void UpdateOutputShape() override;
        virtual void ComputeInternal() override;
        virtual void ComputeGradientInternal(const Tensor& grad) override;

    private:
        float m_Power = 0;
    };
    
    static Operation* pow(TensorLike* x, TensorLike* p, const string& name = "")
    {
        return new PowOp(x, p, name);
    }

    static Operation* square(TensorLike* x, const string& name = "")
    {
        return new PowOp(x, 2.f, name.empty() ? "square" : name);
    }
}
