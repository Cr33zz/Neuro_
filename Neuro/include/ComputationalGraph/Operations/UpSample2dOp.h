#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    class UpSample2dOp : public Operation
    {
    public:
        UpSample2dOp(TensorLike* x, int scaleFactor, const string& name = "");

    protected:
        virtual void UpdateOutputShape() override;
        virtual void ComputeInternal() override;
        virtual void ComputeGradientInternal(const Tensor& grad) override;

    private:
        int m_ScaleFactor;
    };

    static Operation* upsample2d(TensorLike* x, int scaleFactor, const string& name = "")
    {
        return new UpSample2dOp(x, scaleFactor, name);
    }
}
