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
            parallel_for((uint)0, (uint)t1Values.size(), [&](uint i)
            {
                resultValues[i] = alpha * t1Values[i] + beta * t2Values[i];
            });
            return;
        }

        parallel_for((uint)0, t1.Batch(), [&](uint n)
        {
            for (uint i = 0, idx = n * t1.BatchLength(); i < t1.BatchLength(); ++i, ++idx)
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

        parallel_for((uint)0, result.Batch(), [&](uint n)
        {
            uint t1N = min(n, t1Temp.Batch() - 1);
            uint t2N = min(n, t2Temp.Batch() - 1);

            parallel_for((uint)0, t1Temp.Depth(), [&](uint d)
            {
                for (uint h = 0; h < t1Temp.Height(); ++h)
                for (uint w = 0; w < t2Temp.Width(); ++w)
                for (uint i = 0; i < t1Temp.Width(); ++i)
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
            parallel_for((uint)0, (uint)t1Values.size(), [&](uint i)
            {
                resultValues[i] = t1Values[i] * t2Values[i];
            });
            return;
        }

        parallel_for((uint)0, t1.Batch(), [&](uint n)
        {
            for (uint i = 0, idx = n * t1.BatchLength(); i < t1.BatchLength(); ++i, ++idx)
                resultValues[idx] = t1Values[idx] * t2Values[i];
        });
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpMultiCpu::Sum(const Tensor& input, EAxis axis, int batch, Tensor& result) const
    {
        result.Zero();
        input.CopyToHost();
        result.OverrideHost();

        auto& inputValues = input.GetValues();
        auto& resultValues = result.GetValues();

        if (axis == EAxis::Sample)
        {
            uint batchMin = batch < 0 ? 0 : batch;
            uint batchMax = batch < 0 ? input.Batch() : (batch + 1);
            uint batchLen = input.BatchLength();

            parallel_for(batchMin, batchMax, [&](uint n) {
            for (uint i = 0, idx = n * batchLen; i < batchLen; ++i, ++idx)
                resultValues[n - batchMin] += inputValues[idx];
            });
        }
        else if (axis == EAxis::Feature)
        {
            uint batchLen = input.BatchLength();

            parallel_for((uint)0, input.BatchLength(), [&](uint f) {
            for (uint n = 0; n < input.Batch(); ++n)
                resultValues[f] += inputValues[f + n * input.BatchLength()];
            });
        }
        else //if (axis == EAxis::Global)
        {
            for (uint i = 0; i < input.Length(); ++i)
                resultValues[0] += inputValues[i];
        }
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpMultiCpu::Transpose(const Tensor& input, Tensor& result) const
    {
        input.CopyToHost();
        result.OverrideHost();

        parallel_for((uint)0, input.Batch(), [&](uint n) {
        parallel_for((uint)0, input.Depth(), [&](uint d) {
        for (uint h = 0; h < input.Height(); ++h)
        for (uint w = 0; w < input.Width(); ++w)
            result(h, w, d, n) = input(w, h, d, n);
        });
        });
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpMultiCpu::Conv2D(const Tensor& input, const Tensor& kernels, uint stride, uint paddingX, uint paddingY, Tensor& result) const
    {
        input.CopyToHost();
        kernels.CopyToHost();
        result.OverrideHost();

        parallel_for(0, (int)input.Batch(), [&](int n) {
        parallel_for(0, (int)kernels.Batch(), [&](int outD) {
        for (int h = -(int)paddingY, outH = 0; outH < (int)result.Height(); h += (int)stride, ++outH)
        for (int w = -(int)paddingX, outW = 0; outW < (int)result.Width(); w += (int)stride, ++outW)
        {
            float val = 0;

            for (int kernelD = 0; kernelD < (int)kernels.Depth(); ++kernelD)
            for (int kernelH = 0; kernelH < (int)kernels.Height(); ++kernelH)
            for (int kernelW = 0; kernelW < (int)kernels.Width(); ++kernelW)
                val += input.TryGet(0, w + kernelW, h + kernelH, kernelD, n) * kernels(kernelW, kernelH, kernelD, outD);

            result(outW, outH, outD, n) = val;
        }
        });
        });
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpMultiCpu::Conv2DInputGradient(const Tensor& gradient, const Tensor& kernels, uint stride, uint paddingX, uint paddingY, Tensor& inputGradients) const
    {
        gradient.CopyToHost();
        kernels.CopyToHost();
        inputGradients.CopyToHost();

        parallel_for(0, (int)gradient.Batch(), [&](int outN) {
        for (int outD = 0; outD < (int)gradient.Depth(); ++outD)
        for (int outH = 0, h = -(int)paddingY; outH < (int)gradient.Height(); h += (int)stride, ++outH)
        for (int outW = 0, w = -(int)paddingX; outW < (int)gradient.Width(); w += (int)stride, ++outW)
        {
            float chainGradient = gradient.Get(outW, outH, outD, outN);

            for (int kernelH = 0; kernelH < (int)kernels.Height(); ++kernelH)
            {
                int inH = h + kernelH;
                for (int kernelW = 0; kernelW < (int)kernels.Width(); ++kernelW)
                {
                    int inW = w + kernelW;
                    if (inH >= 0 && inH < (int)inputGradients.Height() && inW >= 0 && inW < (int)inputGradients.Width())
                    {
                        for (int kernelD = 0; kernelD < (int)kernels.Depth(); ++kernelD)
                            inputGradients(inW, inH, kernelD, outN) += kernels.Get(kernelW, kernelH, kernelD, outD) * chainGradient;
                    }
                }
            }
        }});
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpMultiCpu::Conv2DKernelsGradient(const Tensor& input, const Tensor& gradient, uint stride, uint paddingX, uint paddingY, Tensor& kernelsGradient) const
    {
        input.CopyToHost();
        gradient.CopyToHost();
        kernelsGradient.CopyToHost();

        parallel_for(0, (int)gradient.Depth(), [&](int outD) {
        for (int outN = 0; outN < (int)gradient.Batch(); ++outN)
        for (int outH = 0, h = -(int)paddingY; outH < (int)gradient.Height(); h += (int)stride, ++outH)
        for (int outW = 0, w = -(int)paddingX; outW < (int)gradient.Width(); w += (int)stride, ++outW)
        {
            float chainGradient = gradient.Get(outW, outH, outD, outN);

            for (int kernelH = 0; kernelH < (int)kernelsGradient.Height(); ++kernelH)
            {
                int inH = h + kernelH;
                for (int kernelW = 0; kernelW < (int)kernelsGradient.Width(); ++kernelW)
                {
                    int inW = w + kernelW;
                    if (inH >= 0 && inH < (int)input.Height() && inW >= 0 && inW < (int)input.Width())
                    {
                        for (int kernelD = 0; kernelD < (int)kernelsGradient.Depth(); ++kernelD)
                            kernelsGradient(kernelW, kernelH, kernelD, outD) += input.Get(inW, inH, kernelD, outN) * chainGradient;
                    }
                }
            }
        }});
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpMultiCpu::Pool2D(const Tensor& input, uint filterSize, uint stride, EPoolingMode type, uint paddingX, uint paddingY, Tensor& output) const
    {
        input.CopyToHost();
        output.OverrideHost();

        parallel_for(0, (int)input.Batch(), [&](int outN) {
        parallel_for(0, (int)input.Depth(), [&](int outD) {
        for (int outH = 0, h = -(int)paddingY; outH < (int)output.Height(); h += (int)stride, ++outH)
        for (int outW = 0, w = -(int)paddingX; outW < (int)output.Width(); w += (int)stride, ++outW)
        {
            if (type == EPoolingMode::Max)
            {
                float value = -numeric_limits<float>().max();

				for (int poolY = 0; poolY < (int)filterSize; ++poolY)
				for (int poolX = 0; poolX < (int)filterSize; ++poolX)
					value = max(value, input.TryGet(-numeric_limits<float>().max(), w + poolX, h + poolY, outD, outN));

				output(outW, outH, outD, outN) = value;
            }
            else if (type == EPoolingMode::Avg)
            {
                float sum = 0;
				for (int poolY = 0; poolY < (int)filterSize; ++poolY)
                for (int poolX = 0; poolX < (int)filterSize; ++poolX)
                    sum += input.TryGet(0, w + poolX, h + poolY, outD, outN);

				output(outW, outH, outD, outN) = sum / (filterSize * filterSize);
            }
        }
        });
        });
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpMultiCpu::Pool2DGradient(const Tensor& output, const Tensor& input, const Tensor& outputGradient, uint filterSize, uint stride, EPoolingMode type, uint paddingX, uint paddingY, Tensor& result) const
    {
        output.CopyToHost();
        input.CopyToHost();
        outputGradient.CopyToHost();
        result.OverrideHost();

        result.Zero();

        parallel_for(0, (int)output.Batch(), [&](int outN) {
        parallel_for(0, (int)output.Depth(), [&](int outD) {
        for (int outH = 0, h = -(int)paddingY; outH < (int)output.Height(); ++outH, h += (int)stride)
        for (int outW = 0, w = -(int)paddingX; outW < (int)output.Width(); ++outW, w += (int)stride)
        {
            if (type == EPoolingMode::Max)
            {
                // use 1 for all elements equal to max value in each pooled matrix and 0 for all others
                for (int poolH = 0; poolH < (int)filterSize; ++poolH)
                for (int poolW = 0; poolW < (int)filterSize; ++poolW)
                {
                    float value = input.TryGet(-numeric_limits<float>::max(), w + poolW, h + poolH, outD, outN);
                    if (value == output(outW, outH, outD, outN))
                        result.TrySet(result.TryGet(-numeric_limits<float>::max(), w + poolW, h + poolH, outD, outN) + outputGradient(outW, outH, outD, outN), w + poolW, h + poolH, outD, outN);
                }
            }
            else if (type == EPoolingMode::Avg)
            {
                float filterElementsNum = (float)filterSize * filterSize;

                for (int poolH = 0; poolH < (int)filterSize; ++poolH)
                for (int poolW = 0; poolW < (int)filterSize; ++poolW)
                {
                    result.TrySet(result.TryGet(-numeric_limits<float>::max(), w + poolW, h + poolH, outD, outN) + outputGradient(outW, outH, outD, outN) / filterElementsNum, w + poolW, h + poolH, outD, outN);
                }
            }
        }
        });
        });
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpMultiCpu::UpSample2D(const Tensor& t, uint scaleFactor, Tensor& result) const
    {
        parallel_for((uint)0, t.Batch(), [&](uint n) {
        parallel_for((uint)0, t.Depth(), [&](uint d) {
        for (uint h = 0; h < t.Height(); ++h)
        for (uint w = 0; w < t.Width(); ++w)
        {
            for (uint outH = h * scaleFactor; outH < (h + 1) * scaleFactor; ++outH)
            for (uint outW = w * scaleFactor; outW < (w + 1) * scaleFactor; ++outW)
                result(outW, outH, d, n) = t(w, h, d, n);
        }
        });
        });
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpMultiCpu::UpSample2DGradient(const Tensor& outputGradient, uint scaleFactor, Tensor& result) const
    {
        parallel_for((uint)0, outputGradient.Batch(), [&](uint n) {
        parallel_for((uint)0, outputGradient.Depth(), [&](uint d) {
        for (uint h = 0; h < outputGradient.Height(); ++h)
        for (uint w = 0; w < outputGradient.Width(); ++w)
            result(w / scaleFactor, h / scaleFactor, d, n) += outputGradient(w, h, d, n);
        });
        });
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpMultiCpu::Map(const function<float(float)>& func, const Tensor& input, Tensor& result) const
    {
        input.CopyToHost();
        result.OverrideHost();

        auto& inputValues = input.GetValues();
        auto& resultValues = result.GetValues();

        parallel_for(0, (int)inputValues.size(), [&](int i)
        {
            resultValues[i] = func(inputValues[i]);
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
            parallel_for((uint)0, (uint)t1Values.size(), [&](uint i)
            {
                resultValues[i] = func(t1Values[i], t2Values[i]);
            });
            return;
        }

        parallel_for((uint)0, t1.Batch(), [&](uint n)
        {
            for (uint i = 0, idx = n * t1.BatchLength(); i < t1.BatchLength(); ++i, ++idx)
                resultValues[idx] = func(t1Values[idx], t2Values[i]);
        });
    }
}
