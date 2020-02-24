#pragma once

#include "Tensors/TensorOpCpu.h"

namespace Neuro
{
    class TensorOpCpuMkl : public TensorOpCpu
    {
    public:
        TensorOpCpuMkl();

        virtual void MatMul(bool transposeT1, bool transposeT2, const Tensor& t1, const Tensor& t2, Tensor& output) const override;
        virtual void Transpose(const Tensor& input, Tensor& output) const override;
    };
}