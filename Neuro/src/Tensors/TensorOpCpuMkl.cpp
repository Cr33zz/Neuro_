#include <mkl.h>

#include "Tensors/TensorOpCpuMkl.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    TensorOpCpuMkl::TensorOpCpuMkl()
    {

    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpCpuMkl::MatMul(bool transposeT1, bool transposeT2, const Tensor& t1, const Tensor& t2, Tensor& output) const
    {
        t1.CopyToHost();
        t2.CopyToHost();
        output.OverrideHost();
        output.Zero();

        int m = t1.Height(), n = t2.Width(), k = t1.Width();
        int lda = transposeT1 ? m : k;
        int ldb = transposeT2 ? k : n;
        int ldc = n;


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