#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    class FuseSubTensorsOp : public Operation
    {
    public:
        FuseSubTensorsOp(const vector<TensorLike*>& xs, size_t tX, size_t tY, const string& name = "");

    protected:
        virtual void UpdateOutputShape() override;
        virtual void ComputeInternal() override;
        virtual void ComputeGradientInternal(const Tensor& grad) override;

    private:
        size_t m_TX;
        size_t m_TY;
    };

    static Operation* fuse_subtensors(const vector<TensorLike*>& xs, size_t tX, size_t tY, const string& name = "")
    {
        return new FuseSubTensorsOp(xs, tX, tY, name);
    }
}
