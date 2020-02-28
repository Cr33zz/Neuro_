#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    class ExtractSubTensorOp : public Operation
    {
    public:
        ExtractSubTensorOp(TensorLike* x, uint32_t width, uint32_t height, uint32_t widthOffset, uint32_t heightOffset, const string& name = "");

    protected:
        virtual void ComputeInternal() override;
        virtual void ComputeGradientInternal(const Tensor& grad) override;
        virtual void UpdateOutputShape() override;

    private:
        uint32_t m_Width;
        uint32_t m_Height;
        uint32_t m_WidthOffset;
        uint32_t m_HeightOffset;
    };

    static Operation* extract_subtensor(TensorLike* x, uint32_t width, uint32_t height, uint32_t widthOffset, uint32_t heightOffset, const string& name = "")
    {
        return new ExtractSubTensorOp(x, width, height, widthOffset, heightOffset, name);
    }
}