#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    class NEURO_DLL_EXPORT FuseSubTensorsOp : public Operation
    {
    public:
        FuseSubTensorsOp(const vector<TensorLike*>& xs, size_t tX, size_t tY, const string& name = "");
        FuseSubTensorsOp(const vector<TensorLike*>& xs, size_t tX, size_t tY, const Shape& outputShape, const string& name = "");

    protected:
        virtual void UpdateOutputShape() override;
        virtual void ComputeInternal() override;
        virtual void ComputeGradientInternal(const Tensor& grad) override;

    private:
        size_t m_TX;
        size_t m_TY;
        bool m_ClampAllowed = false;
    };

    static Operation* fuse_subtensors(const vector<TensorLike*>& xs, size_t tX, size_t tY, const string& name = "")
    {
        return new FuseSubTensorsOp(xs, tX, tY, name);
    }

    static Operation* fuse_subtensors(const vector<TensorLike*>& xs, size_t tX, size_t tY, const Shape& outputShape, const string& name = "")
    {
        return new FuseSubTensorsOp(xs, tX, tY, outputShape, name);
    }
}
