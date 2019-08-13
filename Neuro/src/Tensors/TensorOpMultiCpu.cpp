#include <ppl.h>

#include "Tensors/TensorOpMultiCpu.h"

namespace Neuro
{
    using namespace concurrency;

    //////////////////////////////////////////////////////////////////////////
    void TensorOpMultiCpu::Mul(bool transposeT1, bool transposeT2, const Tensor& t1, const Tensor& t2, Tensor& result) const
    {
        auto t1Temp = transposeT1 ? t1.Transposed() : t1;
        auto t2Temp = transposeT2 ? t2.Transposed() : t2;

        t1Temp.CopyToHost();
        t2Temp.CopyToHost();
        result.Zero();

        parallel_for(0, result.BatchSize(), [&](int n)
        {
            int t1N = min(n, t1Temp.BatchSize() - 1);
            int t2N = min(n, t2Temp.BatchSize() - 1);

            parallel_for(0, t1Temp.Depth(), [&](int d)
            {
                for (int h = 0; h < t1Temp.Height(); ++h)
                    for (int w = 0; w < t2Temp.Width(); ++w)
                        for (int i = 0; i < t1Temp.Width(); ++i)
                            result(w, h, d, n) += t1Temp.Get(i, h, d, t1N) *
                            t2Temp.Get(w, i, d, t2N);
            });
        });
    }
}
