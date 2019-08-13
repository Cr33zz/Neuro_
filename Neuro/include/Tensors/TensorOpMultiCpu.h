#pragma once

#include "Tensors/TensorOpCpu.h"

namespace Neuro
{
    class TensorOpMultiCpu : public TensorOpCpu
    {
    public:
        /*virtual void Add(float alpha, Tensor t1, float beta, Tensor t2, Tensor result) override
        {
            t1.CopyToHost();
            t2.CopyToHost();
            result.set Tensor.Location.Host;

            if (t2.BatchSize == t1.BatchSize)
            {
                var rangePartitioner = Partitioner.Create(0, t1.Values.Length);
                Parallel.ForEach(rangePartitioner, range =>
                {
                    for (int i = range.Item1; i < range.Item2; ++i)
                        result.Values[i] = alpha * t1.Values[i] + beta * t2.Values[i];
                });
                return;
            }

            var rangePartitioner2 = Partitioner.Create(0, t1.BatchLength);

            for (int n = 0; n < t1.BatchSize; ++n)
            {
                Parallel.ForEach(rangePartitioner2, range =>
                {
                    for (int i = range.Item1, idx = n * t1.BatchLength + range.Item1; i < range.Item2; ++i, ++idx)
                        result.Values[idx] = alpha * t1.Values[idx] + beta * t2.Values[i];
                });
            }
        }*/

        virtual void Mul(bool transposeT1, bool transposeT2, const Tensor& t1, const Tensor& t2, Tensor& result) const override;

        //public override void MulElem(Tensor t1, Tensor t2, Tensor result)
        //{
        //    t1.CopyToHost();
        //    t2.CopyToHost();
        //    result.CurrentLocation = Tensor.Location.Host;

        //    var rangePartitioner = Partitioner.Create(0, t1.Values.Length);
        //    Parallel.ForEach(rangePartitioner, range =>
        //    {
        //        for (int i = range.Item1; i < range.Item2; ++i)
        //            result.Values[i] = t1.Values[i] * t2.Values[i];
        //    });
        //}

        //public override void Transpose(Tensor t, Tensor result)
        //{
        //    t.CopyToHost();
        //    result.CurrentLocation = Tensor.Location.Host;

        //    Parallel.For(0, t.BatchSize, n => {
        //        Parallel.For(0, t.Depth, d =>
        //        {
        //            for (int h = 0; h < t.Height; ++h)
        //            for (int w = 0; w < t.Width; ++w)
        //                result[h, w, d, n] = t[w, h, d, n];
        //        });
        //    });
        //}

        //public override void Conv2D(Tensor t, Tensor kernels, int stride, Tensor.PaddingType padding, Tensor result)
        //{
        //    t.CopyToHost();
        //    kernels.CopyToHost();
        //    result.CurrentLocation = Tensor.Location.Host;

        //    int outputWidth = 0, outputHeight = 0, paddingX = 0, paddingY = 0;
        //    Tensor.GetPaddingParams(padding, t.Width, t.Height, kernels.Width, kernels.Height, stride, out outputHeight, out outputWidth, out paddingX, out paddingY);

        //    Parallel.For(0, t.BatchSize, n =>
        //    {
        //        Parallel.For(0, kernels.BatchSize, outD => {
        //        for (int h = -paddingY, outH = 0; outH < result.Height; h += stride, ++outH)
        //        for (int w = -paddingX, outW = 0; outW < result.Width; w += stride, ++outW)
        //        {
        //            float val = 0;

        //            for (int kernelD = 0; kernelD < kernels.Depth; ++kernelD)
        //            for (int kernelH = 0; kernelH < kernels.Height; ++kernelH)
        //            for (int kernelW = 0; kernelW < kernels.Width; ++kernelW)
        //                val += t.TryGet(0, w + kernelW, h + kernelH, kernelD, n) *
        //                       kernels[kernelW, kernelH, kernelD, outD];

        //            result[outW, outH, outD, n] = val;
        //        }});
        //    });
        //}

        //public override void Conv2DInputGradient(Tensor gradient, Tensor kernels, int stride, Tensor.PaddingType padding, Tensor inputGradients)
        //{
        //    gradient.CopyToHost();
        //    kernels.CopyToHost();
        //    inputGradients.CopyToHost();

        //    Tensor rotKernels = kernels.Rotated180();
        //    padding = Tensor.PaddingType.Full;

        //    int outputWidth = 0, outputHeight = 0, paddingX = 0, paddingY = 0;
        //    Tensor.GetPaddingParams(padding, gradient.Width, gradient.Height, kernels.Width, kernels.Height, stride, out outputHeight, out outputWidth, out paddingX, out paddingY);

        //    Parallel.For(0, gradient.BatchSize, n =>
        //    {
        //        for (int outH = 0, h = -paddingY; outH < inputGradients.Height; h += stride, ++outH)
        //        for (int outW = 0, w = -paddingX; outW < inputGradients.Width; w += stride, ++outW)
        //        Parallel.For(0, inputGradients.Depth, outD =>
        //        {
        //            for (int kernelN = 0; kernelN < rotKernels.BatchSize; ++kernelN)
        //            for (int kernelH = 0; kernelH < rotKernels.Height; ++kernelH)
        //            for (int kernelW = 0; kernelW < rotKernels.Width; ++kernelW)
        //            {
        //                inputGradients[outW, outH, outD, n] += gradient.TryGet(0, w + kernelW, h + kernelH, kernelN, n) * rotKernels[kernelW, kernelH, outD, kernelN];
        //            }
        //        });
        //    });
        //}

        //public override void Conv2DKernelsGradient(Tensor input, Tensor gradient, int stride, Tensor.PaddingType padding, Tensor kernelsGradient)
        //{
        //    input.CopyToHost();
        //    gradient.CopyToHost();
        //    kernelsGradient.CopyToHost();

        //    int outputWidth = 0, outputHeight = 0, paddingX = 0, paddingY = 0;
        //    Tensor.GetPaddingParams(padding, input.Width, input.Height, kernelsGradient.Width, kernelsGradient.Height, stride, out outputHeight, out outputWidth, out paddingX, out paddingY);

        //    for (int n = 0; n < gradient.BatchSize; ++n)
        //    {
        //        Parallel.For(0, kernelsGradient.BatchSize, outD =>
        //        {
        //            for (int h = -paddingY, outH = 0; outH < gradient.Height; h += stride, ++outH)
        //            for (int w = -paddingX, outW = 0; outW < gradient.Width; w += stride, ++outW)
        //            {
        //                float grad = gradient[outW, outH, outD, n];

        //                for (int kernelD = 0; kernelD < kernelsGradient.Depth; ++kernelD)
        //                for (int kernelH = 0; kernelH < kernelsGradient.Height; ++kernelH)
        //                for (int kernelW = 0; kernelW < kernelsGradient.Width; ++kernelW)
        //                {
        //                    float kernGradVal = input.TryGet(0, w + kernelW, h + kernelH, kernelD, n) * grad;
        //                    kernelsGradient[kernelW, kernelH, kernelD, outD] += kernGradVal;
        //                }
        //            }
        //        });
        //    }
        //}

        //public override void Pool(Tensor t, int filterSize, int stride, Tensor.PoolType type, int paddingX, int paddingY, Tensor result)
        //{
        //    t.CopyToHost();
        //    result.CurrentLocation = Tensor.Location.Host;

        //    Parallel.For(0, t.BatchSize, outN => 
        //    {
        //        Parallel.For(0, t.Depth, outD =>
        //        {
        //            for (int outH = 0, h = -paddingY; outH < result.Height; h += stride, ++outH)
        //            for (int outW = 0, w = -paddingX; outW < result.Width; w += stride, ++outW)
        //            {
        //                if (type == Tensor.PoolType.Max)
        //                {
        //                    float value = float.MinValue;

        //                    for (int poolY = 0; poolY < filterSize; ++poolY)
        //                    for (int poolX = 0; poolX < filterSize; ++poolX)
        //                    {
        //                        value = Math.Max(value, t.TryGet(float.MinValue, w + poolX, h + poolY, outD, outN));
        //                    }

        //                    result[outW, outH, outD, outN] = value;
        //                }
        //                else if (type == Tensor.PoolType.Avg)
        //                {
        //                    float sum = 0;
        //                    for (int poolY = 0; poolY < filterSize; ++poolY)
        //                    for (int poolX = 0; poolX < filterSize; ++poolX)
        //                        sum += t.TryGet(0, w + poolX, h + poolY, outD, outN);

        //                    result[outW, outH, outD, outN] = sum / (filterSize * filterSize);
        //                }
        //            }
        //        });
        //    });
        //}

        //public override void PoolGradient(Tensor output, Tensor input, Tensor outputGradient, int filterSize, int stride, Tensor.PoolType type, int paddingX, int paddingY, Tensor result)
        //{
        //    output.CopyToHost();
        //    input.CopyToHost();
        //    outputGradient.CopyToHost();
        //    result.CurrentLocation = Tensor.Location.Host;

        //    result.Zero();

        //    Parallel.For(0, output.BatchSize, outN =>
        //    {
        //        Parallel.For(0, output.Depth, outD =>
        //        {
        //            for (int outH = 0, h = -paddingY; outH < output.Height; ++outH, h += stride)
        //            for (int outW = 0, w = -paddingX; outW < output.Width; ++outW, w += stride)
        //            {
        //                if (type == Tensor.PoolType.Max)
        //                {
        //                    // use 1 for all elements equal to max value in each pooled matrix and 0 for all others
        //                    for (int poolH = 0; poolH < filterSize; ++poolH)
        //                    for (int poolW = 0; poolW < filterSize; ++poolW)
        //                    {
        //                        float value = input.TryGet(Single.MinValue, w + poolW, h + poolH, outD, outN);
        //                        if (value == output[outW, outH, outD, outN])
        //                            result.TrySet(result.TryGet(Single.MinValue, w + poolW, h + poolH, outD, outN) + outputGradient[outW, outH, outD, outN], w + poolW, h + poolH, outD, outN);
        //                    }
        //                }
        //                else if (type == Tensor.PoolType.Avg)
        //                {
        //                    float filterElementsNum = filterSize * filterSize;

        //                    for (int poolH = 0; poolH < filterSize; ++poolH)
        //                    for (int poolW = 0; poolW < filterSize; ++poolW)
        //                    {
        //                        result.TrySet(result.TryGet(Single.MinValue, w + poolW, h + poolH, outD, outN) + outputGradient[outW, outH, outD, outN] / filterElementsNum, w + poolW, h + poolH, outD, outN);
        //                    }
        //                }
        //            }
        //        });
        //    });
        //}

        //public override void Map(Func<float, float> func, Tensor t, Tensor result)
        //{
        //    t.CopyToHost();
        //    result.CurrentLocation = Tensor.Location.Host;

        //    var rangePartitioner = Partitioner.Create(0, result.Values.Length);
        //    Parallel.ForEach(rangePartitioner, range =>
        //    {
        //        for (int i = range.Item1; i < range.Item2; ++i)
        //            result.Values[i] = func(t.Values[i]);
        //    });
        //}

        //public override void Map(Func<float, float, float> func, Tensor t1, Tensor t2, Tensor result)
        //{
        //    t1.CopyToHost();
        //    t2.CopyToHost();
        //    result.CurrentLocation = Tensor.Location.Host;

        //    var rangePartitioner = Partitioner.Create(0, result.Values.Length);
        //    Parallel.ForEach(rangePartitioner, range =>
        //    {
        //        for (int i = range.Item1; i < range.Item2; ++i)
        //            result.Values[i] = func(t1.Values[i], t2.Values[i]);
        //    });
        //}

        //public override void SumBatches(Tensor t, Tensor result)
        //{
        //    t.CopyToHost();
        //    result.CurrentLocation = Tensor.Location.Host;

        //    int batchLen = t.BatchLength;

        //    Parallel.For(0, result.BatchSize, n =>
        //    {
        //        for (int i = 0, idx = n * batchLen; i < batchLen; ++i, ++idx)
        //            result.Values[i] += t.Values[idx];
        //    });
        //}
    };
}
