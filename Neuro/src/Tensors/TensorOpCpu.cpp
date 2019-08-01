#include "Tensors/TensorOpCpu.h"
#include "Tensors/Tensor.h"

namespace Neuro
{
	//////////////////////////////////////////////////////////////////////////
	void Neuro::TensorOpCpu::Add(float alpha, const Tensor& t1, float beta, const Tensor& t2, Tensor& result)
	{
		t1.CopyToHost();
		t2.CopyToHost();
		result.CurrentLocation = Tensor.Location.Host;

		if (t2.BatchSize == t1.BatchSize)
		{
			for (int i = 0; i < t1.Values.Length; ++i)
				result.Values[i] = alpha * t1.Values[i] + beta * t2.Values[i];
			return;
		}

		for (int n = 0; n < t1.BatchSize; ++n)
			for (int i = 0, idx = n * t1.BatchLength; i < t1.BatchLength; ++i, ++idx)
				result.Values[idx] = alpha * t1.Values[idx] + beta * t2.Values[i];
	}

	//////////////////////////////////////////////////////////////////////////
	void TensorOpCpu::Sub(const Tensor& t1, const Tensor& t2, Tensor& result)
	{
		Add(1, t1, -1, t2, result);
	}

	//////////////////////////////////////////////////////////////////////////
	void TensorOpCpu::Mul(bool transposeT1, bool transposeT2, const Tensor& t1, const Tensor& t2, Tensor& result)
	{
		var t1Temp = transposeT1 ? t1.Transposed() : t1;
		var t2Temp = transposeT2 ? t2.Transposed() : t2;

		t1Temp.CopyToHost();
		t2Temp.CopyToHost();
		result.Zero();

		int N = t1Temp.Height;
		int M = t2Temp.Width;
		int K = t1Temp.Width;

		for (int n = 0; n < result.BatchSize; ++n)
		{
			int t1N = Math.Min(n, t1Temp.BatchSize - 1);
			int t2N = Math.Min(n, t2Temp.BatchSize - 1);

			for (int d = 0; d < t1Temp.Depth; ++d)
				for (int i = 0; i < N; ++i)
					for (int j = 0; j < M; ++j)
						for (int k = 0; k < K; ++k)
							result[j, i, d, n] += t1Temp[k, i, d, t1N] * t2Temp[j, k, d, t2N];
		}
	}

	//////////////////////////////////////////////////////////////////////////
	void TensorOpCpu::MulElem(const Tensor& t1, const Tensor& t2, Tensor& result)
	{
		t1.CopyToHost();
		t2.CopyToHost();
		result.CurrentLocation = Tensor.Location.Host;

		for (int i = 0; i < t1.Values.Length; ++i)
			result.Values[i] = t1.Values[i] * t2.Values[i];
	}

	//////////////////////////////////////////////////////////////////////////
	void TensorOpCpu::Transpose(Tensor t, Tensor& result)
	{
		t.CopyToHost();
		result.CurrentLocation = Tensor.Location.Host;

		for (int n = 0; n < t.BatchSize; ++n)
			for (int d = 0; d < t.Depth; ++d)
				for (int h = 0; h < t.Height; ++h)
					for (int w = 0; w < t.Width; ++w)
						result[h, w, d, n] = t[w, h, d, n];
	}

	//////////////////////////////////////////////////////////////////////////
	void TensorOpCpu::Map(const function<float(float)>& func, const Tensor& t, Tensor& result)
	{
		t.CopyToHost();
		result.CurrentLocation = Tensor.Location.Host;

		for (int i = 0; i < t.Values.Length; ++i)
			result.Values[i] = func(t.Values[i]);
	}

	//////////////////////////////////////////////////////////////////////////
	void TensorOpCpu::Map(const function<float(float, float)>& func, const Tensor& t1, const Tensor& t2, Tensor& result)
	{
		t1.CopyToHost();
		t2.CopyToHost();
		result.CurrentLocation = Tensor.Location.Host;

		for (int i = 0; i < t1.Values.Length; ++i)
			result.Values[i] = func(t1.Values[i], t2.Values[i]);
	}

	//////////////////////////////////////////////////////////////////////////
	void TensorOpCpu::SumBatches(Tensor t, Tensor& result)
	{
		t.CopyToHost();
		result.CurrentLocation = Tensor.Location.Host;

		int batchLen = t.BatchLength;

		for (int n = 0; n < t.BatchSize; ++n)
			for (int i = 0, idx = n * batchLen; i < batchLen; ++i, ++idx)
				result.Values[i] += t.Values[idx];
	}

	//////////////////////////////////////////////////////////////////////////
	void TensorOpCpu::Elu(const Tensor& input, float alpha, Tensor& result)
	{
		input.Map(x = > x >= 0 ? x : alpha * ((float)Math.Exp(x) - 1), result);
	}

	//////////////////////////////////////////////////////////////////////////
	void TensorOpCpu::EluGradient(const Tensor& output, const Tensor& outputGradient, float alpha, Tensor& result)
	{
		output.Map((x, x2) = > (x > 0 ? 1 : (x + alpha)) * x2, outputGradient, result);
	}

	//////////////////////////////////////////////////////////////////////////
	void TensorOpCpu::Softmax(const Tensor& input, Tensor& result)
	{
		input.CopyToHost();
		result.CurrentLocation = Tensor.Location.Host;

		Tensor shifted = input.Sub(input.Max());
		Tensor exps = shifted.Map(x = > (float)Math.Exp(x));

		for (int n = 0; n < input.BatchSize; ++n)
		{
			float sum = exps.Sum(n);

			for (int d = 0; d < input.Depth; ++d)
				for (int h = 0; h < input.Height; ++h)
					for (int w = 0; w < input.Width; ++w)
						result[w, h, d, n] = exps[w, h, d, n] / sum;
		}
	}

	//////////////////////////////////////////////////////////////////////////
	void TensorOpCpu::SoftmaxGradient(const Tensor& output, const Tensor& outputGradient, Tensor& result)
	{
		output.CopyToHost();
		outputGradient.CopyToHost();
		result.CurrentLocation = Tensor.Location.Host;

		var outputReshaped = output.Reshaped(new Shape(1, Shape.Auto, 1, output.BatchSize));
		Tensor jacob = outputReshaped.DiagFlat().Sub(outputReshaped.Mul(outputReshaped.Transposed()));
		jacob.Mul(outputGradient, result);
	}

	//////////////////////////////////////////////////////////////////////////
	void TensorOpCpu::Conv2D(const Tensor& t, const Tensor& kernels, int stride, Tensor::EPaddingType padding, Tensor& result)
	{
		t.CopyToHost();
		kernels.CopyToHost();
		result.CurrentLocation = Tensor.Location.Host;

		int outputWidth = 0, outputHeight = 0, paddingX = 0, paddingY = 0;
		Tensor.GetPaddingParams(padding, t.Width, t.Height, kernels.Width, kernels.Height, stride, out outputHeight, out outputWidth, out paddingX, out paddingY);

		for (int n = 0; n < t.BatchSize; ++n)
		{
			for (int outD = 0; outD < kernels.BatchSize; ++outD)
				for (int h = -paddingY, outH = 0; outH < result.Height; h += stride, ++outH)
					for (int w = -paddingX, outW = 0; outW < result.Width; w += stride, ++outW)
					{
						float val = 0;

						for (int kernelD = 0; kernelD < kernels.Depth; ++kernelD)
							for (int kernelH = 0; kernelH < kernels.Height; ++kernelH)
								for (int kernelW = 0; kernelW < kernels.Width; ++kernelW)
									val += t.TryGet(0, w + kernelW, h + kernelH, kernelD, n) * kernels[kernelW, kernelH, kernelD, outD];

						result[outW, outH, outD, n] = val;
					}
		}
	}

	//////////////////////////////////////////////////////////////////////////
	void TensorOpCpu::Conv2DInputGradient(const Tensor& gradient, const Tensor& kernels, int stride, Tensor::EPaddingType padding, Tensor inputGradients)
	{
		gradient.CopyToHost();
		kernels.CopyToHost();
		inputGradients.CopyToHost();

		Tensor rotKernels = kernels.Rotated180();
		padding = Tensor.PaddingType.Full;

		int outputWidth = 0, outputHeight = 0, paddingX = 0, paddingY = 0;
		Tensor.GetPaddingParams(padding, gradient.Width, gradient.Height, kernels.Width, kernels.Height, stride, out outputHeight, out outputWidth, out paddingX, out paddingY);

		for (int n = 0; n < gradient.BatchSize; ++n)
		{
			for (int outH = 0, h = -paddingY; outH < inputGradients.Height; h += stride, ++outH)
				for (int outW = 0, w = -paddingX; outW < inputGradients.Width; w += stride, ++outW)
					for (int outD = 0; outD < inputGradients.Depth; ++outD)
					{
						for (int kernelN = 0; kernelN < rotKernels.BatchSize; ++kernelN)
							for (int kernelH = 0; kernelH < rotKernels.Height; ++kernelH)
								for (int kernelW = 0; kernelW < rotKernels.Width; ++kernelW)
								{
									inputGradients[outW, outH, outD, n] += gradient.TryGet(0, w + kernelW, h + kernelH, kernelN, n) * rotKernels[kernelW, kernelH, outD, kernelN];
								}
					}
		}
	}

	//////////////////////////////////////////////////////////////////////////
	void TensorOpCpu::Conv2DKernelsGradient(const Tensor& input, const Tensor& gradient, int stride, Tensor::EPoolType padding, Tensor kernelsGradient)
	{
		input.CopyToHost();
		gradient.CopyToHost();
		kernelsGradient.CopyToHost();

		int outputWidth = 0, outputHeight = 0, paddingX = 0, paddingY = 0;
		Tensor.GetPaddingParams(padding, input.Width, input.Height, kernelsGradient.Width, kernelsGradient.Height, stride, out outputHeight, out outputWidth, out paddingX, out paddingY);

		for (int kernelD = 0; kernelD < kernelsGradient.Depth; ++kernelD)
			for (int kernelH = 0; kernelH < kernelsGradient.Height; ++kernelH)
				for (int kernelW = 0; kernelW < kernelsGradient.Width; ++kernelW)
					for (int kernelN = 0; kernelN < kernelsGradient.BatchSize; ++kernelN)
					{
						for (int outN = 0; outN < gradient.BatchSize; ++outN)
							for (int h = -paddingY, outH = 0; outH < gradient.Height; h += stride, ++outH)
								for (int w = -paddingX, outW = 0; outW < gradient.Width; w += stride, ++outW)
								{
									float grad = gradient[outW, outH, kernelN, outN];
									float kernGradVal = input.TryGet(0, w + kernelW, h + kernelH, kernelD, outN) * grad;
									kernelsGradient[kernelW, kernelH, kernelD, kernelN] += kernGradVal;

									//if (kernelsGradient.Shape.GetIndex(kernelW, kernelH, kernelD, kernelN) == 0)
									//{
									//    Trace.WriteLine($"cid={outN * output.Height * output.Width + outH * output.Width + outW} - {kernGradVal}");
									//}
								}
					}
	}

	//////////////////////////////////////////////////////////////////////////
	void TensorOpCpu::Conv2DGradient_old(const Tensor& input, const Tensor& kernels, const Tensor& outputGradient, int stride, int paddingX, int paddingY, const Tensor& inputGradient, Tensor kernelsGradient)
	{
		input.CopyToHost();
		kernels.CopyToHost();
		outputGradient.CopyToHost();
		inputGradient.CopyToHost();
		kernelsGradient.CopyToHost();

		for (var n = 0; n < input.BatchSize; n++)
		{
			for (var depth = 0; depth < outputGradient.Depth; depth++)
			{
				var y = -paddingY;
				for (var ay = 0; ay < outputGradient.Height; y += stride, ay++)
				{
					var x = -paddingX;
					for (var ax = 0; ax < outputGradient.Width; x += stride, ax++)
					{
						// convolve centered at this particular location
						var chainGradient = outputGradient.Get(ax, ay, depth, n);

						// gradient from above, from chain rule
						for (var fy = 0; fy < kernels.Height; fy++)
						{
							var oy = y + fy; // coordinates in the original input array coordinates
							for (var fx = 0; fx < kernels.Width; fx++)
							{
								var ox = x + fx;
								if (oy >= 0 && oy < input.Height && ox >= 0 && ox < input.Width)
								{
									for (var fd = 0; fd < kernels.Depth; fd++)
									{
										kernelsGradient.Set(kernelsGradient.Get(fx, fy, fd, depth) + input.Get(ox, oy, fd, n) * chainGradient, fx, fy, fd, depth);
										inputGradient.Set(inputGradient.Get(ox, oy, fd, n) + kernels.Get(fx, fy, fd, depth) * chainGradient, ox, oy, fd, n);
									}
								}
							}
						}
					}
				}
			}
		}

		//for (int n = 0; n < output.Batches; ++n)
		//{
		//    for (int outD = 0; outD < kernels.Batches; ++outD)
		//    for (int outH = 0, h = -paddingY; outH < output.Height; h += stride, ++outH)
		//    for (int outW = 0, w = -paddingX; outW < output.Width; w += stride, ++outW)
		//    {
		//        float grad = gradient[outW, outH, outD, n];

		//        for (int kernelD = 0; kernelD < kernels.Depth; ++kernelD)
		//        for (int kernelH = 0; kernelH < kernels.Height; ++kernelH)
		//        for (int kernelW = 0; kernelW < kernels.Width; ++kernelW)
		//        {
		//            float inputGradVal = kernels[kernelW, kernelH, kernelD, outD] * grad;
		//            inputGradient.TrySet(inputGradient.TryGet(0, w + kernelW, h + kernelH, kernelD, n) + inputGradVal, w + kernelW, h + kernelH, kernelD, n);

		//            float kernGradVal = input.TryGet(0, w + kernelW, h + kernelH, kernelD, n) * grad;
		//            kernelsGradient[kernelW, kernelH, kernelD, outD] += kernGradVal;
		//        }
		//    }
		//}
	}

	//////////////////////////////////////////////////////////////////////////
	void TensorOpCpu::Pool(const Tensor& t, int filterSize, int stride, Tensor::EPoolType type, int paddingX, int paddingY, Tensor& result)
	{
		t.CopyToHost();
		result.CurrentLocation = Tensor.Location.Host;

		for (int outN = 0; outN < t.BatchSize; ++outN)
			for (int outD = 0; outD < t.Depth; ++outD)
				for (int outH = 0, h = -paddingY; outH < result.Height; h += stride, ++outH)
					for (int outW = 0, w = -paddingX; outW < result.Width; w += stride, ++outW)
					{
						if (type == Tensor.PoolType.Max)
						{
							float value = float.MinValue;

							for (int poolY = 0; poolY < filterSize; ++poolY)
								for (int poolX = 0; poolX < filterSize; ++poolX)
								{
									value = Math.Max(value, t.TryGet(float.MinValue, w + poolX, h + poolY, outD, outN));
								}

							result[outW, outH, outD, outN] = value;
						}
						else if (type == Tensor.PoolType.Avg)
						{
							float sum = 0;
							for (int poolY = 0; poolY < filterSize; ++poolY)
								for (int poolX = 0; poolX < filterSize; ++poolX)
									sum += t.TryGet(0, w + poolX, h + poolY, outD, outN);

							result[outW, outH, outD, outN] = sum / (filterSize * filterSize);
						}
					}
	}

	//////////////////////////////////////////////////////////////////////////
	void TensorOpCpu::PoolGradient(const Tensor& output, const Tensor& input, const Tensor& outputGradient, int filterSize, int stride, Tensor.PoolType type, int paddingX, int paddingY, Tensor& result)
	{
		output.CopyToHost();
		input.CopyToHost();
		outputGradient.CopyToHost();
		result.CurrentLocation = Tensor.Location.Host;

		result.Zero();

		for (int outN = 0; outN < output.BatchSize; ++outN)
			for (int outD = 0; outD < output.Depth; ++outD)
				for (int outH = 0, h = -paddingY; outH < output.Height; ++outH, h += stride)
					for (int outW = 0, w = -paddingX; outW < output.Width; ++outW, w += stride)
					{
						if (type == Tensor.PoolType.Max)
						{
							for (int poolH = 0; poolH < filterSize; ++poolH)
								for (int poolW = 0; poolW < filterSize; ++poolW)
								{
									float value = input.TryGet(Single.MinValue, w + poolW, h + poolH, outD, outN);
									if (value == output[outW, outH, outD, outN])
										result.TrySet(result.TryGet(Single.MinValue, w + poolW, h + poolH, outD, outN) + outputGradient[outW, outH, outD, outN], w + poolW, h + poolH, outD, outN);
								}
						}
						else if (type == Tensor.PoolType.Avg)
						{
							float filterElementsNum = filterSize * filterSize;

							for (int poolH = 0; poolH < filterSize; ++poolH)
								for (int poolW = 0; poolW < filterSize; ++poolW)
								{
									result.TrySet(result.TryGet(Single.MinValue, w + poolW, h + poolH, outD, outN) + outputGradient[outW, outH, outD, outN] / filterElementsNum, w + poolW, h + poolH, outD, outN);
								}
						}
					}
	}

}

