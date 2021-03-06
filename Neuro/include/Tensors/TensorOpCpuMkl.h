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

        //virtual void Add(float alpha, const Tensor& t1, float beta, const Tensor& t2, Tensor& output) const override;
        //virtual void Mul(float alpha, const Tensor& t1, float beta, const Tensor& t2, Tensor& output) const override;
        //virtual void Mul(const Tensor& input, float v, Tensor& output) const override;
        //virtual void Scale(Tensor& input, float v) const override;
        virtual void MatMul(const Tensor& t1, bool transposeT1, const Tensor& t2, bool transposeT2, Tensor& output) const override;
        virtual void MatMul(const Tensor& t, bool transpose, Tensor& output) const override;
        virtual void Transpose(const Tensor& input, Tensor& output) const override;
#endif
    };
}