#include <ppl.h>

#include "Tensors/TensorOpMultiCpu.h"

namespace Neuro
{
    using namespace concurrency;

    //////////////////////////////////////////////////////////////////////////
    void TensorOpMultiCpu::Add(float alpha, const Tensor& t1, float beta, const Tensor& t2, Tensor& result) const
    {
        t1.CopyToHost();
        t2.CopyToHost();
        result.OverrideHost();

        auto& t1Values = t1.GetValues();
        auto& t2Values = t2.GetValues();
        auto& resultValues = result.GetValues();

        if (t2.Batch() == t1.Batch())
        {
            parallel_for(0, (int)t1Values.size(), [&](int i)
            {
                resultValues[i] = alpha * t1Values[i] + beta * t2Values[i];
            });
            return;
        }

        parallel_for(0, t1.Batch(), [&](int n)
        {
            for (int i = 0, idx = n * t1.BatchLength(); i < t1.BatchLength(); ++i, ++idx)
                resultValues[idx] = alpha * t1Values[idx] + beta * t2Values[i];
        });
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpMultiCpu::Mul(bool transposeT1, bool transposeT2, const Tensor& t1, const Tensor& t2, Tensor& result) const
    {
        auto t1Temp = transposeT1 ? t1.Transposed() : t1;
        auto t2Temp = transposeT2 ? t2.Transposed() : t2;

        t1Temp.CopyToHost();
        t2Temp.CopyToHost();
        result.Zero();

        parallel_for(0, result.Batch(), [&](int n)
        {
            int t1N = min(n, t1Temp.Batch() - 1);
            int t2N = min(n, t2Temp.Batch() - 1);

            parallel_for(0, t1Temp.Depth(), [&](int d)
            {
                for (int h = 0; h < t1Temp.Height(); ++h)
                for (int w = 0; w < t2Temp.Width(); ++w)
                for (int i = 0; i < t1Temp.Width(); ++i)
                    result(w, h, d, n) += t1Temp.Get(i, h, d, t1N) * t2Temp.Get(w, i, d, t2N);
            });
        });
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpMultiCpu::MulElem(const Tensor& t1, const Tensor& t2, Tensor& result) const
    {
        t1.CopyToHost();
        t2.CopyToHost();
        result.OverrideHost();

        auto& t1Values = t1.GetValues();
        auto& t2Values = t2.GetValues();
        auto& resultValues = result.GetValues();

        if (t2.Batch() == t1.Batch())
        {
            parallel_for(0, (int)t1Values.size(), [&](int i)
            {
                resultValues[i] = t1Values[i] * t2Values[i];
            });
            return;
        }

        parallel_for(0, t1.Batch(), [&](int n)
        {
            for (int i = 0, idx = n * t1.BatchLength(); i < t1.BatchLength(); ++i, ++idx)
                resultValues[idx] = t1Values[idx] * t2Values[i];
        });
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpMultiCpu::Transpose(const Tensor& input, Tensor& result) const
    {
        input.CopyToHost();
        result.OverrideHost();

        parallel_for(0, input.Batch(), [&](int n)
        {
            parallel_for(0, input.Depth(), [&](int d)
            {
                for (int h = 0; h < input.Height(); ++h)
                for (int w = 0; w < input.Width(); ++w)
                    result(h, w, d, n) = input(w, h, d, n);
            });
        });
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpMultiCpu::Conv2D(const Tensor& input, const Tensor& kernels, int stride, int paddingX, int paddingY, Tensor& result) const
    {
        input.CopyToHost();
        kernels.CopyToHost();
        result.OverrideHost();

        parallel_for(0, input.Batch(), [&](int n)
        {
            parallel_for(0, kernels.Batch(), [&](int outD)
            {
                for (int h = -paddingY, outH = 0; outH < result.Height(); h += stride, ++outH)
                for (int w = -paddingX, outW = 0; outW < result.Width(); w += stride, ++outW)
                {
                    float val = 0;

                    for (int kernelD = 0; kernelD < kernels.Depth(); ++kernelD)
                    for (int kernelH = 0; kernelH < kernels.Height(); ++kernelH)
                    for (int kernelW = 0; kernelW < kernels.Width(); ++kernelW)
                        val += input.TryGet(0, w + kernelW, h + kernelH, kernelD, n) * kernels(kernelW, kernelH, kernelD, outD);

                    result(outW, outH, outD, n) = val;
                }
            });
        });
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpMultiCpu::Conv2DInputGradient(const Tensor& gradient, const Tensor& kernels, int stride, int paddingX, int paddingY, Tensor& inputGradients) const
    {
        gradient.CopyToHost();
        kernels.CopyToHost();
        inputGradients.CopyToHost();

        parallel_for(0, gradient.Batch(), [&](int outN) {
        for (int outD = 0; outD < gradient.Depth(); ++outD)
        for (int outH = 0, h = -paddingY; outH < gradient.Height(); h += stride, ++outH)
        for (int outW = 0, w = -paddingX; outW < gradient.Width(); w += stride, ++outW)
        {
            float chainGradient = gradient.Get(outW, outH, outD, outN);

            for (int kernelH = 0; kernelH < kernels.Height(); ++kernelH)
            {
                int inH = h + kernelH;
                for (int kernelW = 0; kernelW < kernels.Width(); ++kernelW)
                {
                    int inW = w + kernelW;
                    if (inH >= 0 && inH < inputGradients.Height() && inW >= 0 && inW < inputGradients.Width())
                    {
                        for (int kernelD = 0; kernelD < kernels.Depth(); ++kernelD)
                            inputGradients(inW, inH, kernelD, outN) += kernels.Get(kernelW, kernelH, kernelD, outD) * chainGradient;
                    }
                }
            }
        }});
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpMultiCpu::Conv2DKernelsGradient(const Tensor& input, const Tensor& gradient, int stride, int paddingX, int paddingY, Tensor& kernelsGradient) const
    {
        input.CopyToHost();
        gradient.CopyToHost();
        kernelsGradient.CopyToHost();

        parallel_for(0, gradient.Depth(), [&](int outD) {
        for (int outN = 0; outN < gradient.Batch(); ++outN)
        for (int outH = 0, h = -paddingY; outH < gradient.Height(); h += stride, ++outH)
        for (int outW = 0, w = -paddingX; outW < gradient.Width(); w += stride, ++outW)
        {
            float chainGradient = gradient.Get(outW, outH, outD, outN);

            for (int kernelH = 0; kernelH < kernelsGradient.Height(); ++kernelH)
            {
                int inH = h + kernelH;
                for (int kernelW = 0; kernelW < kernelsGradient.Width(); ++kernelW)
                {
                    int inW = w + kernelW;
                    if (inH >= 0 && inH < input.Height() && inW >= 0 && inW < input.Width())
                    {
                        for (int kernelD = 0; kernelD < kernelsGradient.Depth(); ++kernelD)
                            kernelsGradient(kernelW, kernelH, kernelD, outD) += input.Get(inW, inH, kernelD, outN) * chainGradient;
                    }
                }
            }
        }});
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpMultiCpu::Pool2D(const Tensor& input, int filterSize, int stride, EPoolingMode type, int paddingX, int paddingY, Tensor& output) const
    {
        input.CopyToHost();
        output.OverrideHost();

        parallel_for(0, input.Batch(), [&](int outN) {
        parallel_for(0, input.Depth(), [&](int outD) {
        for (int outH = 0, h = -paddingY; outH < output.Height(); h += stride, ++outH)
        for (int outW = 0, w = -paddingX; outW < output.Width(); w += stride, ++outW)
        {
            if (type == EPoolingMode::Max)
            {
                float value = -numeric_limits<float>().max();

				for (int poolY = 0; poolY < filterSize; ++poolY)
				for (int poolX = 0; poolX < filterSize; ++poolX)
					value = max(value, input.TryGet(-numeric_limits<float>().max(), w + poolX, h + poolY, outD, outN));

				output(outW, outH, outD, outN) = value;
            }
            else if (type == EPoolingMode::Avg)
            {
                float sum = 0;
				for (int poolY = 0; poolY < filterSize; ++poolY)
                for (int poolX = 0; poolX < filterSize; ++poolX)
                    sum += input.TryGet(0, w + poolX, h + poolY, outD, outN);

				output(outW, outH, outD, outN) = sum / (filterSize * filterSize);
            }
        }
        });
        });
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpMultiCpu::Pool2DGradient(const Tensor& output, const Tensor& input, const Tensor& outputGradient, int filterSize, int stride, EPoolingMode type, int paddingX, int paddingY, Tensor& result) const
    {
        output.CopyToHost();
        input.CopyToHost();
        outputGradient.CopyToHost();
        result.OverrideHost();

        result.Zero();

        parallel_for(0, output.Batch(), [&](int outN) {
        parallel_for(0, output.Depth(), [&](int outD) {
        for (int outH = 0, h = -paddingY; outH < output.Height(); ++outH, h += stride)
        for (int outW = 0, w = -paddingX; outW < output.Width(); ++outW, w += stride)
        {
            if (type == EPoolingMode::Max)
            {
                // use 1 for all elements equal to max value in each pooled matrix and 0 for all others
                for (int poolH = 0; poolH < filterSize; ++poolH)
                for (int poolW = 0; poolW < filterSize; ++poolW)
                {
                    float value = input.TryGet(-numeric_limits<float>::max(), w + poolW, h + poolH, outD, outN);
                    if (value == output(outW, outH, outD, outN))
                        result.TrySet(result.TryGet(-numeric_limits<float>::max(), w + poolW, h + poolH, outD, outN) + outputGradient(outW, outH, outD, outN), w + poolW, h + poolH, outD, outN);
                }
            }
            else if (type == EPoolingMode::Avg)
            {
                float filterElementsNum = (float)filterSize * filterSize;

                for (int poolH = 0; poolH < filterSize; ++poolH)
                for (int poolW = 0; poolW < filterSize; ++poolW)
                {
                    result.TrySet(result.TryGet(-numeric_limits<float>::max(), w + poolW, h + poolH, outD, outN) + outputGradient(outW, outH, outD, outN) / filterElementsNum, w + poolW, h + poolH, outD, outN);
                }
            }
        }
        });
        });
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpMultiCpu::UpSample2D(const Tensor& t, int scaleFactor, Tensor& result) const
    {
        parallel_for(0, t.Batch(), [&](int n) {
        parallel_for(0, t.Depth(), [&](int d) {
        for (int h = 0; h < t.Height(); ++h)
        for (int w = 0; w < t.Width(); ++w)
        {
            for (int outH = h * scaleFactor; outH < (h + 1) * scaleFactor; ++outH)
            for (int outW = w * scaleFactor; outW < (w + 1) * scaleFactor; ++outW)
                result(outW, outH, d, n) = t(w, h, d, n);
        }
        });
        });
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpMultiCpu::UpSample2DGradient(const Tensor& outputGradient, int scaleFactor, Tensor& result) const
    {
        parallel_for(0, outputGradient.Batch(), [&](int n) {
        parallel_for(0, outputGradient.Depth(), [&](int d) {
        for (int h = 0; h < outputGradient.Height(); ++h)
        for (int w = 0; w < outputGradient.Width(); ++w)
            result(w / scaleFactor, h / scaleFactor, d, n) += outputGradient(w, h, d, n);
        });
        });
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpMultiCpu::Map(const function<float(float)>& func, const Tensor& t, Tensor& result) const
    {
        t.CopyToHost();
        result.OverrideHost();

        auto& tValues = t.GetValues();
        auto& resultValues = result.GetValues();

        parallel_for(0, (int)tValues.size(), [&](int i)
        {
            resultValues[i] = func(tValues[i]);
        });
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpMultiCpu::Map(const function<float(float, float)>& func, const Tensor& t1, const Tensor& t2, Tensor& result) const
    {
        t1.CopyToHost();
        t2.CopyToHost();
        result.OverrideHost();

        auto& t1Values = t1.GetValues();
        auto& t2Values = t2.GetValues();
        auto& resultValues = result.GetValues();

        if (t2.Batch() == t1.Batch())
        {
            parallel_for(0, (int)t1Values.size(), [&](int i)
            {
                resultValues[i] = func(t1Values[i], t2Values[i]);
            });
            return;
        }

        parallel_for(0, t1.Batch(), [&](int n)
        {
            for (int i = 0, idx = n * t1.BatchLength(); i < t1.BatchLength(); ++i, ++idx)
                resultValues[idx] = func(t1Values[idx], t2Values[i]);
        });
    }
}
