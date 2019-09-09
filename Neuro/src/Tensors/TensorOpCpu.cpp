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
			for (uint32_t i = 0; i < (int)t1Values.size(); ++i)
				resultValues[i] = alpha * t1Values[i] + beta * t2Values[i];
			return;
		}

		for (uint32_t n = 0; n < t1.Batch(); ++n)
		for (uint32_t i = 0, idx = n * t1.BatchLength(); i < t1.BatchLength(); ++i, ++idx)
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

        uint32_t N = t1Temp.Height();
        uint32_t M = t2Temp.Width();
        uint32_t K = t1Temp.Width();

		for (uint32_t n = 0; n < result.Batch(); ++n)
		{
            uint32_t t1N = min(n, t1Temp.Batch() - 1);
            uint32_t t2N = min(n, t2Temp.Batch() - 1);

			for (uint32_t d = 0; d < t1Temp.Depth(); ++d)
			for (uint32_t i = 0; i < N; ++i)
			for (uint32_t j = 0; j < M; ++j)
			for (uint32_t k = 0; k < K; ++k)
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
            for (uint32_t i = 0; i < (int)t1Values.size(); ++i)
                resultValues[i] = t1Values[i] * t2Values[i];
            return;
        }

        for (uint32_t n = 0; n < t1.Batch(); ++n)
        for (uint32_t i = 0, idx = n * t1.BatchLength(); i < t1.BatchLength(); ++i, ++idx)
            resultValues[idx] = t1Values[idx] * t2Values[i];
	}

    //////////////////////////////////////////////////////////////////////////
    void TensorOpCpu::Sum(const Tensor& input, EAxis axis, int batch, Tensor& result) const
    {
        result.Zero();
        input.CopyToHost();
        result.OverrideHost();

        auto& inputValues = input.GetValues();
        auto& resultValues = result.GetValues();

        if (axis == EAxis::Sample)
        {
            uint32_t batchMin = batch < 0 ? 0 : batch;
            uint32_t batchMax = batch < 0 ? input.Batch() : (batch + 1);
            uint32_t batchLen = input.BatchLength();

            for (uint32_t n = batchMin, outN = 0; n < batchMax; ++n, ++outN)
            for (uint32_t i = 0, idx = n * batchLen; i < batchLen; ++i, ++idx)
                resultValues[outN] += inputValues[idx];
        }
        else if (axis == EAxis::Feature)
        {
            uint32_t batchLen = input.BatchLength();

            for (uint32_t i = 0; i < input.Length(); ++i)
                resultValues[i % batchLen] += inputValues[i];
        }
        else //if (axis == EAxis::Global)
        {
            for (uint32_t i = 0; i < input.Length(); ++i)
                resultValues[0] += inputValues[i];
        }
    }

    //////////////////////////////////////////////////////////////////////////
	void TensorOpCpu::Transpose(const Tensor& input, Tensor& result) const
	{
		input.CopyToHost();
        result.OverrideHost();

		for (uint32_t n = 0; n < input.Batch(); ++n)
		for (uint32_t d = 0; d < input.Depth(); ++d)
		for (uint32_t h = 0; h < input.Height(); ++h)
		for (uint32_t w = 0; w < input.Width(); ++w)
			result(h, w, d, n) = input(w, h, d, n);
	}

	//////////////////////////////////////////////////////////////////////////
	void TensorOpCpu::Map(const function<float(float)>& func, const Tensor& t, Tensor& result) const
	{
		t.CopyToHost();
        result.OverrideHost();

        auto& tValues = t.GetValues();
        auto& resultValues = result.GetValues();

		for (uint32_t i = 0; i < (uint32_t)tValues.size(); ++i)
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
            for (uint32_t i = 0; i < (uint32_t)t1Values.size(); ++i)
                resultValues[i] = func(t1Values[i], t2Values[i]);
            return;
        }

        for (uint32_t n = 0; n < t1.Batch(); ++n)
        for (uint32_t i = 0, idx = n * t1.BatchLength(); i < t1.BatchLength(); ++i, ++idx)
            resultValues[idx] = func(t1Values[idx], t2Values[i]);
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
    void TensorOpCpu::LeakyReLU(const Tensor& input, float alpha, Tensor& output) const
    {
        input.Map([&](float x) { return x >= 0 ? x : (alpha * x); }, output);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpCpu::LeakyReLUGradient(const Tensor& output, const Tensor& outputGradient, float alpha, Tensor& inputGradient) const
    {
        output.Map([&](float x, float x2) { return (x > 0 ? 1 : alpha) * x2; }, outputGradient, inputGradient);
    }

	//////////////////////////////////////////////////////////////////////////
	void TensorOpCpu::Softmax(const Tensor& input, Tensor& result) const
	{
		input.CopyToHost();
        result.OverrideHost();

		Tensor shifted = input.Sub(input.Max(EAxis::Global)(0));
        Tensor exps = shifted.Map([&](float x) { return (float)exp(x); });

        auto& expsValues = exps.GetValues();
        auto& resultValues = result.GetValues();

		for (uint32_t n = 0; n < input.Batch(); ++n)
		{
			float sum = exps.Sum(EAxis::Sample, n)(0);

            for (uint32_t i = 0, idx = n * input.BatchLength(); i < input.BatchLength(); ++i, ++idx)
                resultValues[idx] = expsValues[idx] / sum;
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
	void TensorOpCpu::Conv2D(const Tensor& input, const Tensor& kernels, uint32_t stride, uint32_t paddingX, uint32_t paddingY, Tensor& output) const
	{
		input.CopyToHost();
		kernels.CopyToHost();
        output.OverrideHost();

		for (int n = 0; n < (int)input.Batch(); ++n)
		for (int outD = 0; outD < (int)kernels.Batch(); ++outD)
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
	}

	//////////////////////////////////////////////////////////////////////////
	void TensorOpCpu::Conv2DInputGradient(const Tensor& gradient, const Tensor& kernels, uint32_t stride, uint32_t paddingX, uint32_t paddingY, Tensor& inputGradients) const
	{
		gradient.CopyToHost();
		kernels.CopyToHost();
		inputGradients.CopyToHost();

        for (int outN = 0; outN < (int)gradient.Batch(); ++outN)
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
        }
	}

	//////////////////////////////////////////////////////////////////////////
	void TensorOpCpu::Conv2DKernelsGradient(const Tensor& input, const Tensor& gradient, uint32_t stride, uint32_t paddingX, uint32_t paddingY, Tensor& kernelsGradient) const
	{
		input.CopyToHost();
		gradient.CopyToHost();
		kernelsGradient.CopyToHost();

        for (int outN = 0; outN < (int)gradient.Batch(); ++outN)
        for (int outD = 0; outD < (int)gradient.Depth(); ++outD)
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
        }
	}

	//////////////////////////////////////////////////////////////////////////
	void TensorOpCpu::Pool2D(const Tensor& input, uint32_t filterSize, uint32_t stride, EPoolingMode type, uint32_t paddingX, uint32_t paddingY, Tensor& output) const
	{
		input.CopyToHost();
        output.OverrideHost();

        for (int outN = 0; outN < (int)input.Batch(); ++outN)
		for (int outD = 0; outD < (int)input.Depth(); ++outD)
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
	}

	//////////////////////////////////////////////////////////////////////////
    void TensorOpCpu::Pool2DGradient(const Tensor& output, const Tensor& input, const Tensor& outputGradient, uint32_t filterSize, uint32_t stride, EPoolingMode type, uint32_t paddingX, uint32_t paddingY, Tensor& result) const
	{
		output.CopyToHost();
		input.CopyToHost();
		outputGradient.CopyToHost();
		result.OverrideHost();
		result.Zero();

		for (int outN = 0; outN < (int)output.Batch(); ++outN)
		for (int outD = 0; outD < (int)output.Depth(); ++outD)
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
                            result.TrySet(result.TryGet(-numeric_limits<float>().max(), w + poolW, h + poolH, outD, outN) + outputGradient(outW, outH, outD, outN), w + poolW, h + poolH, outD, outN);
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
                float filterElementsNum = (float)(filterSize * filterSize);

				for (int poolH = 0; poolH < (int)filterSize; ++poolH)
				for (int poolW = 0; poolW < (int)filterSize; ++poolW)
				{
					result.TrySet(result.TryGet(-numeric_limits<float>().max(), w + poolW, h + poolH, outD, outN) + outputGradient(outW, outH, outD, outN) / filterElementsNum, w + poolW, h + poolH, outD, outN);
				}
			}
		}
	}

    //////////////////////////////////////////////////////////////////////////
    void TensorOpCpu::UpSample2D(const Tensor& input, uint32_t scaleFactor, Tensor& output) const
    {
        for (uint32_t n = 0; n < input.Batch(); ++n)
        for (uint32_t d = 0; d < input.Depth(); ++d)
        for (uint32_t h = 0; h < input.Height(); ++h)
        for (uint32_t w = 0; w < input.Width(); ++w)
        {
            for (uint32_t outH = h * scaleFactor; outH < (h + 1) * scaleFactor; ++outH)
            for (uint32_t outW = w * scaleFactor; outW < (w + 1) * scaleFactor; ++outW)
                output(outW, outH, d, n) = input(w, h, d, n);
        }
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpCpu::UpSample2DGradient(const Tensor& outputGradient, uint32_t scaleFactor, Tensor& inputGradient) const
    {
        for (uint32_t n = 0; n < outputGradient.Batch(); ++n)
        for (uint32_t d = 0; d < outputGradient.Depth(); ++d)
        for (uint32_t h = 0; h < outputGradient.Height(); ++h)
        for (uint32_t w = 0; w < outputGradient.Width(); ++w)
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
        saveMean = input.Avg(EAxis::Feature);
        Tensor xmu = input.Sub(saveMean);
        Tensor carre = xmu.Map([](float x) { return x * x; });
        Tensor variance = carre.Avg(EAxis::Feature);
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
        float n = (float)outputGradient.Batch();

        Tensor xmu = input.Sub(savedMean);
        Tensor carre = xmu.Map([](float x) { return x * x; });
        Tensor va2 = xmu.MulElem(savedInvVariance);
        Tensor variance = carre.Avg(EAxis::Feature);
        Tensor sqrtvar = variance.Map([](float x) { return sqrt(x); });

        betaGradient = outputGradient.Sum(EAxis::Feature);
        gammaGradient = va2.MulElem(outputGradient).Sum(EAxis::Feature);

        Tensor dva2 = outputGradient.MulElem(gamma);
        Tensor dxmu = dva2.MulElem(savedInvVariance);
        Tensor dinvvar = xmu.MulElem(dva2).Sum(EAxis::Feature);
        Tensor dsqrtvar = dinvvar.Map([&](float x1, float x2) { return -1.f / (x2*x2) * x1; }, sqrtvar);
        Tensor dvar = dsqrtvar.Map([&](float x1, float x2) { return 0.5f * pow(x2 + _EPSILON, -0.5f) * x1; }, variance);
        Tensor dcarre = Tensor(carre.GetShape()).FillWithValue(1).MulElem(dvar).Mul(1.f / n);
        dxmu.Add(xmu.MulElem(dcarre).Mul(2), dxmu);
        Tensor dmu = dxmu.Sum(EAxis::Feature).Negated();
        dxmu.Add(Tensor(dxmu.GetShape()).FillWithValue(1).MulElem(dmu).Mul(1.f / n), inputGradient);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpCpu::Dropout(const Tensor& input, float prob, Tensor& saveMask, Tensor& output)
    {
        saveMask.FillWithFunc([&]() { return (GlobalRng().NextFloat() < prob ? 0.f : 1.f) / prob; });
        input.MulElem(saveMask, output);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpCpu::DropoutGradient(const Tensor& outputGradient, const Tensor& savedMask, Tensor& inputGradient)
    {
        outputGradient.MulElem(savedMask, inputGradient);
    }
}
