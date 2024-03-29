#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    class NEURO_DLL_EXPORT ExtractSubTensorOp : public Operation
    {
    public:
        ExtractSubTensorOp(TensorLike* x, uint32_t width, uint32_t height, uint32_t widthOffset, uint32_t heightOffset, bool clampAllowed = false, const string& name = "");

    protected:
        virtual void ComputeInternal() override;
        virtual void ComputeGradientInternal(const Tensor& grad) override;
        virtual void UpdateOutputShape() override;

    private:
        uint32_t m_Width;
        uint32_t m_Height;
        uint32_t m_WidthOffset;
        uint32_t m_HeightOffset;
        bool m_ClampAllowed;
    };

    static Operation* extract_subtensor(TensorLike* x, uint32_t width, uint32_t height, uint32_t widthOffset, uint32_t heightOffset, bool clampAllowed = false, const string& name = "")
    {
        return new ExtractSubTensorOp(x, width, height, widthOffset, heightOffset, clampAllowed, name);
    }
}