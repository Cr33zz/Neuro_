#ifndef MKL_DISABLED
#include <mkl.h>

#include "Tensors/TensorOpCpuMkl.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    TensorOpCpuMkl::TensorOpCpuMkl()
    {
    }

    //////////////////////////////////////////////////////////////////////////
    /*void TensorOpCpuMkl::Add(float alpha, const Tensor& t1, float beta, const Tensor& t2, Tensor& output) const
    {
        if (t1.GetShape() != t2.GetShape())
            return __super::Add(alpha, t1, beta, t2, output);

        t1.CopyToHost();
        t2.CopyToHost();
        output.OverrideHost();

        MKL_INT n = t1.Length();
        vsadd(&n, t1.Values(), t2.Values(), output.Values());
    }*/

    //////////////////////////////////////////////////////////////////////////
    /*void TensorOpCpuMkl::Mul(float alpha, const Tensor& t1, float beta, const Tensor& t2, Tensor& output) const
    {
        if (t1.GetShape() != t2.GetShape())
            return __super::Mul(alpha, t1, beta, t2, output);

        t1.CopyToHost();
        t2.CopyToHost();
        output.OverrideHost();

        MKL_INT n = t1.Length();
        vsmul(&n, t1.Values(), t2.Values(), output.Values());
    }*/

    //////////////////////////////////////////////////////////////////////////
    //void TensorOpCpuMkl::Mul(const Tensor& input, float v, Tensor& output) const
    //{
    //    input.CopyToHost();
    //    output.OverrideHost();

    //    mkl_somatcopy(
    //        'r',
    //        'n',
    //        1,
    //        input.Length(),
    //        v,
    //        input.Values(),
    //        input.Length(),
    //        output.Values(),
    //        output.Length());
    //}

    //////////////////////////////////////////////////////////////////////////
    /*void TensorOpCpuMkl::Scale(Tensor& input, float v) const
    {
        input.CopyToHost();

        mkl_simatcopy(
            'r',
            'n',
            1,
            input.Length(),
            v,
            input.Values(),
            input.Length(),
            input.Length());
    }*/

    //////////////////////////////////////////////////////////////////////////
    void TensorOpCpuMkl::MatMul(const Tensor& t1, bool transposeT1, const Tensor& t2, bool transposeT2, Tensor& output) const
    {
        t1.CopyToHost();
        t2.CopyToHost();
        output.OverrideHost();
        output.Zero();

        int m = transposeT1 ? t1.Width() : t1.Height();
        int n = transposeT2 ? t2.Height() : t2.Width();
        int k = transposeT1 ? t1.Height() : t1.Width();
        int lda = t1.Width();
        int ldb = t2.Width();
        int ldc = output.Width();

        float alpha = 1, beta = 0;

        for (uint32_t b = 0; b < output.Batch(); ++b)
        {
            uint32_t t1B = min(b, t1.Batch() - 1);
            uint32_t t2B = min(b, t2.Batch() - 1);

            for (uint32_t d = 0; d < t1.Depth(); ++d)
            {
                cblas_sgemm(
                    CblasRowMajor,
                    transposeT1 ? CblasTrans : CblasNoTrans,
                    transposeT2 ? CblasTrans : CblasNoTrans,
                    m,
                    n,
                    k,
                    alpha,
                    t1.Values() + d * t1.GetShape().Dim0Dim1 + t1B * t1.BatchLength(),
                    lda,
                    t2.Values() + d * t2.GetShape().Dim0Dim1 + t2B * t2.BatchLength(),
                    ldb,
                    beta,
                    output.Values() + d * output.GetShape().Dim0Dim1 + b * output.BatchLength(),
                    ldc);
            }
        }
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpCpuMkl::MatMul(const Tensor& t, bool transpose, Tensor& output) const
    {
        t.CopyToHost();
        output.OverrideHost();

        int n = transpose ? t.Width() : t.Height();
        int k = transpose ? t.Height() : t.Width();
        int lda = t.Width();
        int ldc = n;

        float alpha = 1, beta = 0;

        uint32_t outWidth = output.Width();
        float* outVals = output.Values();

        for (uint32_t b = 0; b < t.Batch(); ++b)
        {
            for (uint32_t d = 0; d < t.Depth(); ++d)
            {
                cblas_ssyrk(
                    CblasRowMajor,
                    CblasLower,
                    transpose ? CblasTrans : CblasNoTrans,
                    n,
                    k,
                    alpha,
                    t.Values() + d * t.GetShape().Dim0Dim1 + b * t.BatchLength(),
                    lda,
                    beta,
                    output.Values() + d * output.GetShape().Dim0Dim1 + b * output.BatchLength(),
                    ldc);

                uint32_t offset = d * t.GetShape().Dim0Dim1 + b * t.BatchLength();

                #pragma omp parallel for
                for (int h = 0; h < (int)output.Height(); ++h)
                for (uint32_t w = h + 1; w < outWidth; ++w)
                    outVals[offset + (uint32_t)h * outWidth + w] = outVals[offset + w * outWidth + (uint32_t)h];
            }
        }
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpCpuMkl::Transpose(const Tensor& input, Tensor& output) const
    {
        input.CopyToHost();
        output.OverrideHost();

        int m = input.Height();
        uint32_t n = input.Width();

        //treat depth as batch
        int batches = input.Depth() * input.Batch();
        float alpha = 1;

        for (int b = 0; b < batches; ++b)
        {
            const float* tPtr = input.Values() + b * input.GetShape().Dim0Dim1;

            mkl_somatcopy(
                'r',
                't',
                input.Height(),
                input.Width(),
                alpha,
                tPtr,
                input.Width(),
                output.Values() + b * output.GetShape().Dim0Dim1,
                input.Height());
        }
    }
}
#endif