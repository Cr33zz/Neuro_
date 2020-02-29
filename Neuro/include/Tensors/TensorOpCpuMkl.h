#pragma once

#include "Tensors/TensorOpCpu.h"

namespace Neuro
{
    class TensorOpCpuMkl : public TensorOpCpu
    {
#ifndef MKL_DISABLED
    public:
        TensorOpCpuMkl();
        virtual EOpMode OpMode() const { return CPU_MKL; }

        virtual void MatMul(bool transposeT1, bool transposeT2, const Tensor& t1, const Tensor& t2, Tensor& output) const override;
        virtual void Transpose(const Tensor& input, Tensor& output) const override;
#endif
    };
}