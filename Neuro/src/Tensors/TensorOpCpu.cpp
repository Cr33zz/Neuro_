#include <algorithm>
#include <functional>

#include "Tools.h"
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
	void TensorOpCpu::Transpose(const Tensor& input, Tensor& result) const
	{
		input.CopyToHost();
        result.OverrideHost();

		for (int n = 0; n < input.Batch(); ++n)
		for (int d = 0; d < input.Depth(); ++d)
		for (int h = 0; h < input.Height(); ++h)
		for (int w = 0; w < input.Width(); ++w)
			result(h, w, d, n) = input(w, h, d, n);
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

        if (t2.Batch() == t1.Batch())
        {
            for (int i = 0; i < (int)t1Values.size(); ++i)
                resultValues[i] = func(t1Values[i], t2Values[i]);
            return;
        }

        for (int n = 0; n < t1.Batch(); ++n)
        for (int i = 0, idx = n * t1.BatchLength(); i < t1.BatchLength(); ++i, ++idx)
            resultValues[idx] = func(t1Values[idx], t2Values[i]);
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
	void TensorOpCpu::Elu(const Tensor& input, float alpha, Tensor& output) const
	{
        input.Map([&](float x) { return x >= 0 ? x : alpha * ((float)exp(x) - 1); }, output);
	}

	//////////////////////////////////////////////////////////////////////////
	void TensorOpCpu::EluGradient(const Tensor& output, const Tensor& outputGradient, float alpha, Tensor& inputGradient) const
	{
        output.Map([&](float x, float x2) { return (x > 0 ? 1 : (x + alpha)) * x2; }, outputGradient, inputGradient);
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
	void TensorOpCpu::SoftmaxGradient(const Tensor& output, const Tensor& outputGradient, Tensor& inputGradient) const
	{
		output.CopyToHost();
		outputGradient.CopyToHost();
        inputGradient.OverrideHost();

		Tensor outputReshaped = output.Reshaped(Shape(1, Shape::Auto, 1, output.Batch()));
		Tensor jacob = outputReshaped.DiagFlat().Sub(outputReshaped.Mul(outputReshaped.Transposed()));
		jacob.Mul(outputGradient, inputGradient);
	}

	//////////////////////////////////////////////////////////////////////////
	void TensorOpCpu::Conv2D(const Tensor& input, const Tensor& kernels, int stride, int paddingX, int paddingY, Tensor& result) const
	{
		input.CopyToHost();
		kernels.CopyToHost();
        result.OverrideHost();

		for (int n = 0; n < input.Batch(); ++n)
		{
			for (int outD = 0; outD < kernels.Batch(); ++outD)
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
		}
	}

	//////////////////////////////////////////////////////////////////////////
	void TensorOpCpu::Conv2DInputGradient(const Tensor& gradient, const Tensor& kernels, int stride, int paddingX, int paddingY, Tensor& inputGradients) const
	{
		gradient.CopyToHost();
		kernels.CopyToHost();
		inputGradients.CopyToHost();

        for (int outN = 0; outN < gradient.Batch(); ++outN)
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
        }
	}

	//////////////////////////////////////////////////////////////////////////
	void TensorOpCpu::Conv2DKernelsGradient(const Tensor& input, const Tensor& gradient, int stride, int paddingX, int paddingY, Tensor& kernelsGradient) const
	{
		input.CopyToHost();
		gradient.CopyToHost();
		kernelsGradient.CopyToHost();

        for (int outN = 0; outN < gradient.Batch(); ++outN)
        for (int outD = 0; outD < gradient.Depth(); ++outD)
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
        }
	}

	//////////////////////////////////////////////////////////////////////////
	void TensorOpCpu::Pool2D(const Tensor& input, int filterSize, int stride, EPoolingMode type, int paddingX, int paddingY, Tensor& output) const
	{
		input.CopyToHost();
        output.OverrideHost();

        for (int outN = 0; outN < input.Batch(); ++outN)
		for (int outD = 0; outD < input.Depth(); ++outD)
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
	}

	//////////////////////////////////////////////////////////////////////////
    void TensorOpCpu::Pool2DGradient(const Tensor& output, const Tensor& input, const Tensor& outputGradient, int filterSize, int stride, EPoolingMode type, int paddingX, int paddingY, Tensor& result) const
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
                    float value = input.TryGet(-numeric_limits<float>().max(), w + poolW, h + poolH, outD, outN);
					if (value == output(outW, outH, outD, outN))
						result.TrySet(result.TryGet(-numeric_limits<float>().max(), w + poolW, h + poolH, outD, outN) + outputGradient(outW, outH, outD, outN), w + poolW, h + poolH, outD, outN);
				}
			}
			else if (type == EPoolingMode::Avg)
			{
				int filterElementsNum = filterSize * filterSize;

				for (int poolH = 0; poolH < filterSize; ++poolH)
				for (int poolW = 0; poolW < filterSize; ++poolW)
				{
					result.TrySet(result.TryGet(-numeric_limits<float>().max(), w + poolW, h + poolH, outD, outN) + outputGradient(outW, outH, outD, outN) / filterElementsNum, w + poolW, h + poolH, outD, outN);
				}
			}
		}
	}

    //////////////////////////////////////////////////////////////////////////
    void TensorOpCpu::UpSample2D(const Tensor& input, int scaleFactor, Tensor& output) const
    {
        for (int n = 0; n < input.Batch(); ++n)
        for (int d = 0; d < input.Depth(); ++d)
        for (int h = 0; h < input.Height(); ++h)
        for (int w = 0; w < input.Width(); ++w)
        {
            for (int outH = h * scaleFactor; outH < (h + 1) * scaleFactor; ++outH)
            for (int outW = w * scaleFactor; outW < (w + 1) * scaleFactor; ++outW)
                output(outW, outH, d, n) = input(w, h, d, n);
        }
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpCpu::UpSample2DGradient(const Tensor& outputGradient, int scaleFactor, Tensor& inputGradient) const
    {
        for (int n = 0; n < outputGradient.Batch(); ++n)
        for (int d = 0; d < outputGradient.Depth(); ++d)
        for (int h = 0; h < outputGradient.Height(); ++h)
        for (int w = 0; w < outputGradient.Width(); ++w)
            inputGradient(w / scaleFactor, h / scaleFactor, d, n) += outputGradient(w, h, d, n);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpCpu::BatchNormalization(const Tensor& input, const Tensor& gamma, const Tensor& beta, const Tensor& runningMean, const Tensor& runningVar, Tensor& output) const
    {
        Tensor xbar = input.Sub(runningMean);
        xbar.Map([&](float x1, float x2) { return x1 / sqrt(x2 + _EPSILON); }, runningVar, xbar);
        xbar.MulElem(gamma).Add(beta, output);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpCpu::BatchNormalizationTrain(const Tensor& input, const Tensor& gamma, const Tensor& beta, float momentum, Tensor& runningMean, Tensor& runningVar, Tensor& saveMean, Tensor& saveInvVariance, Tensor& output) const
    {
        float N = (float)input.Batch();
        saveMean = input.SumBatches().Mul(1.f / N);
        Tensor xmu = input.Sub(saveMean);
        Tensor carre = xmu.Map([](float x) { return x * x; });
        Tensor variance = carre.SumBatches().Mul(1.f / N);
        Tensor sqrtvar = variance.Map([](float x) { return sqrt(x); });
        saveInvVariance = sqrtvar.Map([](float x) { return 1.f / x; });
        Tensor va2 = xmu.MulElem(saveInvVariance);
        va2.MulElem(gamma).Add(beta, output);

        runningMean.Map([&](float x1, float x2) { return momentum * x1 + (1.f - momentum) * x2; }, saveMean, runningMean);
        runningVar.Map([&](float x1, float x2) { return momentum * x1 + (1.f - momentum) * x2; }, variance, runningVar);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpCpu::BatchNormalizationGradient(const Tensor& input, const Tensor& gamma, const Tensor& outputGradient, const Tensor& savedMean, const Tensor& savedInvVariance, Tensor& gammaGradient, Tensor& betaGradient, Tensor& inputGradient) const
    {
        float N = (float)outputGradient.Batch();

        Tensor xmu = input.Sub(savedMean);
        Tensor carre = xmu.Map([](float x) { return x * x; });        
        Tensor va2 = xmu.MulElem(savedInvVariance);
        Tensor variance = carre.SumBatches().Mul(1.f / N);
        Tensor sqrtvar = variance.Map([](float x) { return sqrt(x); });

        betaGradient = outputGradient.SumBatches();
        gammaGradient = va2.MulElem(outputGradient).SumBatches();

        Tensor dva2 = outputGradient.MulElem(gamma);
        Tensor dxmu = dva2.MulElem(savedInvVariance);
        Tensor dinvvar = xmu.MulElem(dva2).SumBatches();
        Tensor dsqrtvar = dinvvar.Map([&](float x1, float x2) { return -1.f / (x2*x2) * x1; }, sqrtvar);
        Tensor dvar = dsqrtvar.Map([&](float x1, float x2) { return 0.5f * pow(x2 + _EPSILON, -0.5f) * x1; }, variance);
        Tensor dcarre = Tensor(carre.GetShape()).FillWithValue(1).MulElem(dvar).Mul(1.f / N);
        dxmu.Add(xmu.MulElem(dcarre).Mul(2), dxmu);
        Tensor dmu = dxmu.SumBatches().Negated();
        dxmu.Add(Tensor(dxmu.GetShape()).FillWithValue(1).MulElem(dmu).Mul(1.f / N), inputGradient);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpCpu::Dropout(const Tensor& input, float prob, Tensor& saveMask, Tensor& output)
    {
        saveMask.FillWithFunc([&]() { return (g_Rng.NextFloat() < prob ? 0.f : 1.f) / prob; });
        input.MulElem(saveMask, output);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpCpu::DropoutGradient(const Tensor& outputGradient, const Tensor& savedMask, Tensor& inputGradient)
    {
        outputGradient.MulElem(savedMask, inputGradient);
    }
}
