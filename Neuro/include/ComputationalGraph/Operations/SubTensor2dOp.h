#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    class SubTensor2dOp : public Operation
    {
    public:
        SubTensor2dOp(TensorLike* x, uint32_t width, uint32_t height, uint32_t widthOffset, uint32_t heightOffset, const string& name = "");

    protected:
        virtual void UpdateOutputShape() override;
        virtual void ComputeInternal() override;
        virtual void ComputeGradientInternal(const Tensor& grad) override;

    private:
        uint32_t m_Width;
        uint32_t m_Height;
        uint32_t m_WidthOffset;
        uint32_t m_HeightOffset;
    };

    static Operation* sub_tensor2d(TensorLike* x, uint32_t width, uint32_t height, uint32_t widthOffset, uint32_t heightOffset, const string& name = "")
    {
        return new SubTensor2dOp(x, width, height, widthOffset, heightOffset, name);
    }
}
