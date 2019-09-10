#include <ppl.h>

#include "Tensors/TensorOpMultiCpu.h"

namespace Neuro
{
    using namespace concurrency;

    //////////////////////////////////////////////////////////////////////////
    void TensorOpMultiCpu::Add(float alpha, const Tensor& t1, float beta, const Tensor& t2, Tensor& output) const
    {
        t1.CopyToHost();
        t2.CopyToHost();
        output.OverrideHost();

        auto& t1Values = t1.GetValues();
        auto& t2Values = t2.GetValues();
        auto& outputValues = output.GetValues();

        if (t2.Batch() == t1.Batch())
        {
            parallel_for((uint32_t)0, (uint32_t)t1Values.size(), [&](uint32_t i)
            {
                outputValues[i] = alpha * t1Values[i] + beta * t2Values[i];
            });
            return;
        }

        parallel_for((uint32_t)0, t1.Batch(), [&](uint32_t n)
        {
            for (uint32_t i = 0, idx = n * t1.BatchLength(); i < t1.BatchLength(); ++i, ++idx)
                outputValues[idx] = alpha * t1Values[idx] + beta * t2Values[i];
        });
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpMultiCpu::Mul(bool transposeT1, bool transposeT2, const Tensor& t1, const Tensor& t2, Tensor& output) const
    {
        auto t1Temp = transposeT1 ? t1.Transposed() : t1;
        auto t2Temp = transposeT2 ? t2.Transposed() : t2;

        t1Temp.CopyToHost();
        t2Temp.CopyToHost();
        output.Zero();

        parallel_for((uint32_t)0, output.Batch(), [&](uint32_t n)
        {
            uint32_t t1N = min(n, t1Temp.Batch() - 1);
            uint32_t t2N = min(n, t2Temp.Batch() - 1);

            parallel_for((uint32_t)0, t1Temp.Depth(), [&](uint32_t d)
            {
                for (uint32_t h = 0; h < t1Temp.Height(); ++h)
                for (uint32_t w = 0; w < t2Temp.Width(); ++w)
                for (uint32_t i = 0; i < t1Temp.Width(); ++i)
                    output(w, h, d, n) += t1Temp.Get(i, h, d, t1N) * t2Temp.Get(w, i, d, t2N);
            });
        });
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpMultiCpu::MulElem(const Tensor& t1, const Tensor& t2, Tensor& output) const
    {
        t1.CopyToHost();
        t2.CopyToHost();
        output.OverrideHost();

        auto& t1Values = t1.GetValues();
        auto& t2Values = t2.GetValues();
        auto& outputValues = output.GetValues();

        if (t2.Batch() == t1.Batch())
        {
            parallel_for((uint32_t)0, (uint32_t)t1Values.size(), [&](uint32_t i)
            {
                outputValues[i] = t1Values[i] * t2Values[i];
            });
            return;
        }

        parallel_for((uint32_t)0, t1.Batch(), [&](uint32_t n)
        {
            for (uint32_t i = 0, idx = n * t1.BatchLength(); i < t1.BatchLength(); ++i, ++idx)
                outputValues[idx] = t1Values[idx] * t2Values[i];
        });
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpMultiCpu::Sum(const Tensor& input, EAxis axis, int batch, Tensor& output) const
    {
        output.Zero();
        input.CopyToHost();
        output.OverrideHost();

        auto& inputValues = input.GetValues();
        auto& outputValues = output.GetValues();

        if (axis == EAxis::Sample)
        {
            uint32_t batchMin = batch < 0 ? 0 : batch;
            uint32_t batchMax = batch < 0 ? input.Batch() : (batch + 1);
            uint32_t batchLen = input.BatchLength();

            parallel_for(batchMin, batchMax, [&](uint32_t n) {
            for (uint32_t i = 0, idx = n * batchLen; i < batchLen; ++i, ++idx)
                outputValues[n - batchMin] += inputValues[idx];
            });
        }
        else if (axis == EAxis::Feature)
        {
            uint32_t batchLen = input.BatchLength();

            parallel_for((uint32_t)0, input.BatchLength(), [&](uint32_t f) {
            for (uint32_t n = 0; n < input.Batch(); ++n)
                outputValues[f] += inputValues[f + n * input.BatchLength()];
            });
        }
        else //if (axis == EAxis::Global)
        {
            for (uint32_t i = 0; i < input.Length(); ++i)
                outputValues[0] += inputValues[i];
        }
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpMultiCpu::Transpose(const Tensor& input, Tensor& output) const
    {
        input.CopyToHost();
        output.OverrideHost();

        parallel_for((uint32_t)0, input.Batch(), [&](uint32_t n) {
        parallel_for((uint32_t)0, input.Depth(), [&](uint32_t d) {
        for (uint32_t h = 0; h < input.Height(); ++h)
        for (uint32_t w = 0; w < input.Width(); ++w)
            output(h, w, d, n) = input(w, h, d, n);
        });
        });
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpMultiCpu::Conv2D(const Tensor& input, const Tensor& kernels, uint32_t stride, uint32_t paddingX, uint32_t paddingY, Tensor& output) const
    {
        input.CopyToHost();
        kernels.CopyToHost();
        output.OverrideHost();

        parallel_for(0, (int)input.Batch(), [&](int n) {
        parallel_for(0, (int)kernels.Batch(), [&](int outD) {
        for (int h = -(int)paddingY, outH = 0; outH < (int)output.Height(); h += (int)stride, ++outH)
        for (int w = -(int)paddingX, outW = 0; outW < (int)output.Width(); w += (int)stride, ++outW)
        {
            float val = 0;

            for (int kernelD = 0; kernelD < (int)kernels.Depth(); ++kernelD)
            for (int kernelH = 0; kernelH < (int)kernels.Height(); ++kernelH)
            for (int kernelW = 0; kernelW < (int)kernels.Width(); ++kernelW)
                val += input.TryGet(0, w + kernelW, h + kernelH, kernelD, n) * kernels(kernelW, kernelH, kernelD, outD);

            output(outW, outH, outD, n) = val;
        }
        });
        });
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpMultiCpu::Conv2DInputGradient(const Tensor& gradient, const Tensor& kernels, uint32_t stride, uint32_t paddingX, uint32_t paddingY, Tensor& inputGradient) const
    {
        gradient.CopyToHost();
        kernels.CopyToHost();
        inputGradient.CopyToHost();

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
                    if (inH >= 0 && inH < (int)inputGradient.Height() && inW >= 0 && inW < (int)inputGradient.Width())
                    {
                        for (int kernelD = 0; kernelD < (int)kernels.Depth(); ++kernelD)
                            inputGradient(inW, inH, kernelD, outN) += kernels.Get(kernelW, kernelH, kernelD, outD) * chainGradient;
                    }
                }
            }
        }});
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpMultiCpu::Conv2DKernelsGradient(const Tensor& input, const Tensor& gradient, uint32_t stride, uint32_t paddingX, uint32_t paddingY, Tensor& kernelsGradient) const
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
    void TensorOpMultiCpu::Pool2D(const Tensor& input, uint32_t filterSize, uint32_t stride, EPoolingMode type, uint32_t paddingX, uint32_t paddingY, Tensor& output) const
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
    void TensorOpMultiCpu::Pool2DGradient(const Tensor& output, const Tensor& input, const Tensor& outputGradient, uint32_t filterSize, uint32_t stride, EPoolingMode type, uint32_t paddingX, uint32_t paddingY, Tensor& inputGradient) const
    {
        output.CopyToHost();
        input.CopyToHost();
        outputGradient.CopyToHost();
        inputGradient.OverrideHost();
        inputGradient.Zero();

        parallel_for(0, (int)output.Batch(), [&](int outN) {
        parallel_for(0, (int)output.Depth(), [&](int outD) {
        for (int outH = 0, h = -(int)paddingY; outH < (int)output.Height(); ++outH, h += (int)stride)
        for (int outW = 0, w = -(int)paddingX; outW < (int)output.Width(); ++outW, w += (int)stride)
        {
            if (type == EPoolingMode::Max)
            {
                bool maxFound = false;
                for (int poolH = 0; poolH < (int)filterSize; ++poolH)
                {
                    for (int poolW = 0; poolW < (int)filterSize; ++poolW)
                    {
                        float value = input.TryGet(-numeric_limits<float>().max(), w + poolW, h + poolH, outD, outN);
                        if (value == output(outW, outH, outD, outN))
                        {
                            inputGradient.TrySet(inputGradient.TryGet(-numeric_limits<float>().max(), w + poolW, h + poolH, outD, outN) + outputGradient(outW, outH, outD, outN), w + poolW, h + poolH, outD, outN);
                            maxFound = true;
                            break;
                        }
                    }

                    if (maxFound)
                        break;
                }
            }
            else if (type == EPoolingMode::Avg)
            {
                float filterElementsNum = (float)filterSize * filterSize;

                for (int poolH = 0; poolH < (int)filterSize; ++poolH)
                for (int poolW = 0; poolW < (int)filterSize; ++poolW)
                {
                    inputGradient.TrySet(inputGradient.TryGet(-numeric_limits<float>::max(), w + poolW, h + poolH, outD, outN) + outputGradient(outW, outH, outD, outN) / filterElementsNum, w + poolW, h + poolH, outD, outN);
                }
            }
        }
        });
        });
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpMultiCpu::UpSample2D(const Tensor& t, uint32_t scaleFactor, Tensor& output) const
    {
        output.CopyToHost();
        output.OverrideHost();
        output.Zero();

        parallel_for((uint32_t)0, t.Batch(), [&](uint32_t n) {
        parallel_for((uint32_t)0, t.Depth(), [&](uint32_t d) {
        for (uint32_t h = 0; h < t.Height(); ++h)
        for (uint32_t w = 0; w < t.Width(); ++w)
        {
            for (uint32_t outH = h * scaleFactor; outH < (h + 1) * scaleFactor; ++outH)
            for (uint32_t outW = w * scaleFactor; outW < (w + 1) * scaleFactor; ++outW)
                output(outW, outH, d, n) = t(w, h, d, n);
        }
        });
        });
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpMultiCpu::UpSample2DGradient(const Tensor& outputGradient, uint32_t scaleFactor, Tensor& inputGradient) const
    {
        outputGradient.CopyToHost();
        inputGradient.OverrideHost();
        inputGradient.Zero();

        parallel_for((uint32_t)0, outputGradient.Batch(), [&](uint32_t n) {
        parallel_for((uint32_t)0, outputGradient.Depth(), [&](uint32_t d) {
        for (uint32_t h = 0; h < outputGradient.Height(); ++h)
        for (uint32_t w = 0; w < outputGradient.Width(); ++w)
            inputGradient(w / scaleFactor, h / scaleFactor, d, n) += outputGradient(w, h, d, n);
        });
        });
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpMultiCpu::Map(const function<float(float)>& func, const Tensor& input, Tensor& output) const
    {
        input.CopyToHost();
        output.OverrideHost();

        auto& inputValues = input.GetValues();
        auto& outputValues = output.GetValues();

        parallel_for(0, (int)inputValues.size(), [&](int i)
        {
            outputValues[i] = func(inputValues[i]);
        });
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpMultiCpu::Map(const function<float(float, float)>& func, const Tensor& t1, const Tensor& t2, Tensor& output) const
    {
        t1.CopyToHost();
        t2.CopyToHost();
        output.OverrideHost();

        auto& t1Values = t1.GetValues();
        auto& t2Values = t2.GetValues();
        auto& outputValues = output.GetValues();

        if (t2.Batch() == t1.Batch())
        {
            parallel_for((uint32_t)0, (uint32_t)t1Values.size(), [&](uint32_t i)
            {
                outputValues[i] = func(t1Values[i], t2Values[i]);
            });
            return;
        }

        parallel_for((uint32_t)0, t1.Batch(), [&](uint32_t n)
        {
            for (uint32_t i = 0, idx = n * t1.BatchLength(); i < t1.BatchLength(); ++i, ++idx)
                outputValues[idx] = func(t1Values[idx], t2Values[i]);
        });
    }
}
