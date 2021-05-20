#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    class NEURO_DLL_EXPORT ClipOp : public Operation
    {
    public:
        ClipOp(TensorLike* x, float min, float max, const string& name = "");

    protected:
        virtual void ComputeInternal() override;
        virtual void ComputeGradientInternal(const Tensor& grad) override;

    private:
        float m_Min;
        float m_Max;
    };

    static Operation* clip(TensorLike* x, float min, float max, const string& name = "")
    {
        return new ClipOp(x, min, max, name);
    }
}
