#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    class SwapRedBlueChannelsOp : public Operation
    {
    public:
        SwapRedBlueChannelsOp(TensorLike* x, const string& name = "");

    protected:
        virtual void ComputeInternal() override;
        virtual void ComputeGradientInternal(const Tensor& grad) override;
    };

    static Operation* swap_red_blue_channels(TensorLike* x, const string& name = "")
    {
        return new SwapRedBlueChannelsOp(x, name);
    }
}
