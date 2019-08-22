#include <algorithm>
#include <functional>
#include "Tensors/TensorOpCpu.h"
#include "Tensors/Tensor.h"

namespace Neuro
{
    using namespace std;

	//////////////////////////////////////////////////////////////////////////
	void Neuro::TensorOpCpu::Add(float alpha, const Tensor& t1, float beta, const Tensor& t2, Tensor& result) const
	{
		t1.CopyToHost();
		t2.CopyToHost();
		result.OverrideHost();

        auto& t1Values = t1.GetValues();
        auto& t2Values = t2.GetValues();
        auto& resultValues = result.GetValues();

		if (t2.Batch() == t1.Batch())
		{
			for (int i = 0; i < (int)t1Values.size(); ++i)
				resultValues[i] = alpha * t1Values[i] + beta * t2Values[i];
			return;
		}

		for (int n = 0; n < t1.Batch(); ++n)
		for (int i = 0, idx = n * t1.BatchLength(); i < t1.BatchLength(); ++i, ++idx)
			resultValues[idx] = alpha * t1Values[idx] + beta * t2Values[i];
	}

	//////////////////////////////////////////////////////////////////////////
	void TensorOpCpu::Sub(const Tensor& t1, const Tensor& t2, Tensor& result) const
	{
		Add(1, t1, -1, t2, result);
	}

	//////////////////////////////////////////////////////////////////////////
	void TensorOpCpu::Mul(bool transposeT1, bool transposeT2, const Tensor& t1, const Tensor& t2, Tensor& result) const
	{
		const Tensor& t1Temp = transposeT1 ? t1.Transposed() : t1;
        const Tensor& t2Temp = transposeT2 ? t2.Transposed() : t2;

		t1Temp.CopyToHost();
		t2Temp.CopyToHost();
		result.Zero();

		int N = t1Temp.Height();
		int M = t2Temp.Width();
		int K = t1Temp.Width();

		for (int n = 0; n < result.Batch(); ++n)
		{
			int t1N = min(n, t1Temp.Batch() - 1);
			int t2N = min(n, t2Temp.Batch() - 1);

			for (int d = 0; d < t1Temp.Depth(); ++d)
			for (int i = 0; i < N; ++i)
			for (int j = 0; j < M; ++j)
			for (int k = 0; k < K; ++k)
				result(j, i, d, n) += t1Temp(k, i, d, t1N) * t2Temp(j, k, d, t2N);
		}
	}

	//////////////////////////////////////////////////////////////////////////
	void TensorOpCpu::MulElem(const Tensor& t1, const Tensor& t2, Tensor& result) const
	{
		t1.CopyToHost();
		t2.CopyToHost();
		result.OverrideHost();

        auto& t1Values = t1.GetValues();
        auto& t2Values = t2.GetValues();
        auto& resultValues = result.GetValues();

        if (t2.Batch() == t1.Batch())
        {
            for (int i = 0; i < (int)t1Values.size(); ++i)
                resultValues[i] = t1Values[i] * t2Values[i];
            return;
        }

        for (int n = 0; n < t1.Batch(); ++n)
        for (int i = 0, idx = n * t1.BatchLength(); i < t1.BatchLength(); ++i, ++idx)
            resultValues[idx] = t1Values[idx] * t2Values[i];
	}

	//////////////////////////////////////////////////////////////////////////
	void TensorOpCpu::Transpose(const Tensor& t, Tensor& result) const
	{
		t.CopyToHost();
        result.OverrideHost();

		for (int n = 0; n < t.Batch(); ++n)
		for (int d = 0; d < t.Depth(); ++d)
		for (int h = 0; h < t.Height(); ++h)
		for (int w = 0; w < t.Width(); ++w)
			result(h, w, d, n) = t(w, h, d, n);
	}

	//////////////////////////////////////////////////////////////////////////
	void TensorOpCpu::Map(const function<float(float)>& func, const Tensor& t, Tensor& result) const
	{
		t.CopyToHost();
        result.OverrideHost();

        auto& tValues = t.GetValues();
        auto& resultValues = result.GetValues();

		for (int i = 0; i < (int)tValues.size(); ++i)
			resultValues[i] = func(tValues[i]);
	}

	//////////////////////////////////////////////////////////////////////////
	void TensorOpCpu::Map(const function<float(float, float)>& func, const Tensor& t1, const Tensor& t2, Tensor& result) const
	{
		t1.CopyToHost();
		t2.CopyToHost();
        result.OverrideHost();

        auto& t1Values = t1.GetValues();
        auto& t2Values = t2.GetValues();
        auto& resultValues = result.GetValues();

		for (int i = 0; i < (int)t1Values.size(); ++i)
			resultValues[i] = func(t1Values[i], t2Values[i]);
	}

	//////////////////////////////////////////////////////////////////////////
	void TensorOpCpu::SumBatches(const Tensor& t, Tensor& result) const
	{
		t.CopyToHost();
        result.OverrideHost();

        auto& tValues = t.GetValues();
        auto& resultValues = result.GetValues();

		int batchLen = t.BatchLength();

		for (int n = 0; n < t.Batch(); ++n)
		for (int i = 0, idx = n * batchLen; i < batchLen; ++i, ++idx)
			resultValues[i] += tValues[idx];
	}

	//////////////////////////////////////////////////////////////////////////
	void TensorOpCpu::Elu(const Tensor& input, float alpha, Tensor& result) const
	{
        input.Map([&](float x) { return x >= 0 ? x : alpha * ((float)exp(x) - 1); }, result);
	}

	//////////////////////////////////////////////////////////////////////////
	void TensorOpCpu::EluGradient(const Tensor& output, const Tensor& outputGradient, float alpha, Tensor& result) const
	{
        output.Map([&](float x, float x2) { return (x > 0 ? 1 : (x + alpha)) * x2; }, outputGradient, result);
	}

	//////////////////////////////////////////////////////////////////////////
	void TensorOpCpu::Softmax(const Tensor& input, Tensor& result) const
	{
		input.CopyToHost();
        result.OverrideHost();

		Tensor shifted = input.Sub(input.Max());
        Tensor exps = shifted.Map([&](float x) { return (float)exp(x); });

		for (int n = 0; n < input.Batch(); ++n)
		{
			float sum = exps.Sum(n);

			for (int d = 0; d < input.Depth(); ++d)
			for (int h = 0; h < input.Height(); ++h)
			for (int w = 0; w < input.Width(); ++w)
				result(w, h, d, n) = exps(w, h, d, n) / sum;
		}
	}

	//////////////////////////////////////////////////////////////////////////
	void TensorOpCpu::SoftmaxGradient(const Tensor& output, const Tensor& outputGradient, Tensor& result) const
	{
		output.CopyToHost();
		outputGradient.CopyToHost();
        result.OverrideHost();

		Tensor outputReshaped = output.Reshaped(Shape(1, Shape::Auto, 1, output.Batch()));
		Tensor jacob = outputReshaped.DiagFlat().Sub(outputReshaped.Mul(outputReshaped.Transposed()));
		jacob.Mul(outputGradient, result);
	}

	//////////////////////////////////////////////////////////////////////////
	void TensorOpCpu::Conv2D(const Tensor& t, const Tensor& kernels, int stride, EPaddingMode padding, Tensor& result) const
	{
		t.CopyToHost();
		kernels.CopyToHost();
        result.OverrideHost();

		int outputWidth = 0, outputHeight = 0, paddingX = 0, paddingY = 0;
		Tensor::GetPaddingParams(padding, t.Width(), t.Height(), kernels.Width(), kernels.Height(), stride, outputHeight, outputWidth, paddingX, paddingY);

		for (int n = 0; n < t.Batch(); ++n)
		{
			for (int outD = 0; outD < kernels.Batch(); ++outD)
			for (int h = -paddingY, outH = 0; outH < result.Height(); h += stride, ++outH)
			for (int w = -paddingX, outW = 0; outW < result.Width(); w += stride, ++outW)
			{
				float val = 0;

				for (int kernelD = 0; kernelD < kernels.Depth(); ++kernelD)
				for (int kernelH = 0; kernelH < kernels.Height(); ++kernelH)
				for (int kernelW = 0; kernelW < kernels.Width(); ++kernelW)
					val += t.TryGet(0, w + kernelW, h + kernelH, kernelD, n) * kernels(kernelW, kernelH, kernelD, outD);

				result(outW, outH, outD, n) = val;
			}
		}
	}

	//////////////////////////////////////////////////////////////////////////
	void TensorOpCpu::Conv2DInputGradient(const Tensor& gradient, const Tensor& kernels, int stride, EPaddingMode padding, Tensor& inputGradients) const
	{
		gradient.CopyToHost();
		kernels.CopyToHost();
		inputGradients.CopyToHost();

		Tensor rotKernels = kernels.Rotated180();

		int outputWidth = 0, outputHeight = 0, paddingX = 0, paddingY = 0;
		Tensor::GetPaddingParams(padding, gradient.Width(), gradient.Height(), kernels.Width(), kernels.Height(), stride, outputHeight, outputWidth, paddingX, paddingY);

		for (int n = 0; n < gradient.Batch(); ++n)
		{
			for (int outH = 0, h = -paddingY; outH < inputGradients.Height(); h += stride, ++outH)
			for (int outW = 0, w = -paddingX; outW < inputGradients.Width(); w += stride, ++outW)
			for (int outD = 0; outD < inputGradients.Depth(); ++outD)
			{
				for (int kernelN = 0; kernelN < rotKernels.Batch(); ++kernelN)
				for (int kernelH = 0; kernelH < rotKernels.Height(); ++kernelH)
				for (int kernelW = 0; kernelW < rotKernels.Width(); ++kernelW)
					inputGradients(outW, outH, outD, n) += gradient.TryGet(0, w + kernelW, h + kernelH, kernelN, n) * rotKernels(kernelW, kernelH, outD, kernelN);
			}
		}
	}

	//////////////////////////////////////////////////////////////////////////
	void TensorOpCpu::Conv2DKernelsGradient(const Tensor& input, const Tensor& gradient, int stride, EPaddingMode padding, Tensor& kernelsGradient) const
	{
		input.CopyToHost();
		gradient.CopyToHost();
		kernelsGradient.CopyToHost();

		int outputWidth = 0, outputHeight = 0, paddingX = 0, paddingY = 0;
		Tensor::GetPaddingParams(padding, input.Width(), input.Height(), kernelsGradient.Width(), kernelsGradient.Height(), stride, outputHeight, outputWidth, paddingX, paddingY);

		for (int kernelD = 0; kernelD < kernelsGradient.Depth(); ++kernelD)
		for (int kernelH = 0; kernelH < kernelsGradient.Height(); ++kernelH)
		for (int kernelW = 0; kernelW < kernelsGradient.Width(); ++kernelW)
		for (int kernelN = 0; kernelN < kernelsGradient.Batch(); ++kernelN)
		{
			for (int outN = 0; outN < gradient.Batch(); ++outN)
			for (int h = -paddingY, outH = 0; outH < gradient.Height(); h += stride, ++outH)
			for (int w = -paddingX, outW = 0; outW < gradient.Width(); w += stride, ++outW)
			{
				float grad = gradient(outW, outH, kernelN, outN);
				float kernGradVal = input.TryGet(0, w + kernelW, h + kernelH, kernelD, outN) * grad;
				kernelsGradient(kernelW, kernelH, kernelD, kernelN) += kernGradVal;
			}
		}
	}

	//////////////////////////////////////////////////////////////////////////
	void TensorOpCpu::Pool(const Tensor& t, int filterSize, int stride, EPoolingMode type, int paddingX, int paddingY, Tensor& result) const
	{
		t.CopyToHost();
        result.OverrideHost();

		for (int outN = 0; outN < t.Batch(); ++outN)
		for (int outD = 0; outD < t.Depth(); ++outD)
		for (int outH = 0, h = -paddingY; outH < result.Height(); h += stride, ++outH)
		for (int outW = 0, w = -paddingX; outW < result.Width(); w += stride, ++outW)
		{
			if (type == EPoolingMode::Max)
			{
				float value = numeric_limits<float>().min();

				for (int poolY = 0; poolY < filterSize; ++poolY)
				for (int poolX = 0; poolX < filterSize; ++poolX)
					value = max(value, t.TryGet(numeric_limits<float>().min(), w + poolX, h + poolY, outD, outN));

				result(outW, outH, outD, outN) = value;
			}
			else if (type == EPoolingMode::Avg)
			{
				float sum = 0;
				for (int poolY = 0; poolY < filterSize; ++poolY)
                for (int poolX = 0; poolX < filterSize; ++poolX)
                    sum += t.TryGet(0, w + poolX, h + poolY, outD, outN);

				result(outW, outH, outD, outN) = sum / (filterSize * filterSize);
			}
		}
	}

	//////////////////////////////////////////////////////////////////////////
    void TensorOpCpu::PoolGradient(const Tensor& output, const Tensor& input, const Tensor& outputGradient, int filterSize, int stride, EPoolingMode type, int paddingX, int paddingY, Tensor& result) const
	{
		output.CopyToHost();
		input.CopyToHost();
		outputGradient.CopyToHost();
		result.OverrideHost();

		result.Zero();

		for (int outN = 0; outN < output.Batch(); ++outN)
		for (int outD = 0; outD < output.Depth(); ++outD)
		for (int outH = 0, h = -paddingY; outH < output.Height(); ++outH, h += stride)
		for (int outW = 0, w = -paddingX; outW < output.Width(); ++outW, w += stride)
		{
			if (type == EPoolingMode::Max)
			{
				for (int poolH = 0; poolH < filterSize; ++poolH)
				for (int poolW = 0; poolW < filterSize; ++poolW)
				{
                    float value = input.TryGet(numeric_limits<float>().min(), w + poolW, h + poolH, outD, outN);
					if (value == output(outW, outH, outD, outN))
						result.TrySet(result.TryGet(numeric_limits<float>().min(), w + poolW, h + poolH, outD, outN) + outputGradient(outW, outH, outD, outN), w + poolW, h + poolH, outD, outN);
				}
			}
			else if (type == EPoolingMode::Avg)
			{
				int filterElementsNum = filterSize * filterSize;

				for (int poolH = 0; poolH < filterSize; ++poolH)
				for (int poolW = 0; poolW < filterSize; ++poolW)
				{
					result.TrySet(result.TryGet(numeric_limits<float>().min(), w + poolW, h + poolH, outD, outN) + outputGradient(outW, outH, outD, outN) / filterElementsNum, w + poolW, h + poolH, outD, outN);
				}
			}
		}
	}
}
