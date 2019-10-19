#include <algorithm>
#include <functional>

#include "Tools.h"
#include "Tensors/TensorOpCpu.h"
#include "Tensors/Tensor.h"

namespace Neuro
{
    using namespace std;

    //////////////////////////////////////////////////////////////////////////
    void TensorOpCpu::Zero(Tensor& input) const
    {
        input.OverrideHost();
        memset(&input.GetValues()[0], 0, input.Length() * sizeof(float));
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpCpu::One(Tensor& input) const
    {
        input.OverrideHost();
        auto& inputValues = input.GetValues();
        fill(inputValues.begin(), inputValues.end(), 1.f);
    }

	//////////////////////////////////////////////////////////////////////////
	void Neuro::TensorOpCpu::Add(float alpha, const Tensor& t1, float beta, const Tensor& t2, Tensor& output) const
	{
		t1.CopyToHost();
		t2.CopyToHost();
		output.OverrideHost();

        for (uint32_t n = 0; n < max(t1.Batch(), t2.Batch()); ++n)
        {
            uint32_t t1N = n % t1.Batch();
            uint32_t t2N = n % t2.Batch();

            for (uint32_t d = 0; d < max(t1.Depth(), t2.Depth()); ++d)
            {
                uint32_t t1D = d % t1.Depth();
                uint32_t t2D = d % t2.Depth();

                for (uint32_t h = 0; h < max(t1.Height(), t2.Height()); ++h)
                {
                    uint32_t t1H = h % t1.Height();
                    uint32_t t2H = h % t2.Height();

                    for (uint32_t w = 0; w < max(t1.Width(), t2.Width()); ++w)
                    {
                        uint32_t t1W = w % t1.Width();
                        uint32_t t2W = w % t2.Width();

                        output(w, h, d, n) = alpha * t1(t1W, t1H, t1D, t1N) + beta * t2(t2W, t2H, t2D, t2N);
                    }
                }
            }
        }
	}

    //////////////////////////////////////////////////////////////////////////
	void TensorOpCpu::Sub(const Tensor& t1, const Tensor& t2, Tensor& output) const
	{
		Add(1, t1, -1, t2, output);
	}

    //////////////////////////////////////////////////////////////////////////
	void TensorOpCpu::MatMul(bool transposeT1, bool transposeT2, const Tensor& t1, const Tensor& t2, Tensor& output) const
	{
		const Tensor& t1Temp = transposeT1 ? t1.Transposed() : t1;
        const Tensor& t2Temp = transposeT2 ? t2.Transposed() : t2;

		t1Temp.CopyToHost();
		t2Temp.CopyToHost();
        output.OverrideHost();
		output.Zero();

        uint32_t N = t1Temp.Height();
        uint32_t M = t2Temp.Width();
        uint32_t K = t1Temp.Width();

		for (uint32_t n = 0; n < output.Batch(); ++n)
		{
            uint32_t t1N = min(n, t1Temp.Batch() - 1);
            uint32_t t2N = min(n, t2Temp.Batch() - 1);

			for (uint32_t d = 0; d < t1Temp.Depth(); ++d)
			for (uint32_t i = 0; i < N; ++i)
			for (uint32_t j = 0; j < M; ++j)
			for (uint32_t k = 0; k < K; ++k)
				output(j, i, d, n) += t1Temp(k, i, d, t1N) * t2Temp(j, k, d, t2N);
		}
	}

	//////////////////////////////////////////////////////////////////////////
	void TensorOpCpu::MulElem(const Tensor& t1, const Tensor& t2, Tensor& output) const
	{
        t1.CopyToHost();
        t2.CopyToHost();
        output.OverrideHost();

        for (uint32_t n = 0; n < max(t1.Batch(), t2.Batch()); ++n)
        {
            uint32_t t1N = n % t1.Batch();
            uint32_t t2N = n % t2.Batch();

            for (uint32_t d = 0; d < max(t1.Depth(), t2.Depth()); ++d)
            {
                uint32_t t1D = d % t1.Depth();
                uint32_t t2D = d % t2.Depth();

                for (uint32_t h = 0; h < max(t1.Height(), t2.Height()); ++h)
                {
                    uint32_t t1H = h % t1.Height();
                    uint32_t t2H = h % t2.Height();

                    for (uint32_t w = 0; w < max(t1.Width(), t2.Width()); ++w)
                    {
                        uint32_t t1W = w % t1.Width();
                        uint32_t t2W = w % t2.Width();

                        output(w, h, d, n) = t1(t1W, t1H, t1D, t1N) * t2(t2W, t2H, t2D, t2N);
                    }
                }
            }
        }
	}

    //////////////////////////////////////////////////////////////////////////
    void TensorOpCpu::Mul(const Tensor& input, float v, Tensor& output) const
    {
        input.CopyToHost();
        output.OverrideHost();

        auto& inputValues = input.GetValues();
        auto& outputValues = output.GetValues();

        for (uint32_t i = 0; i < inputValues.size(); ++i)
            outputValues[i] = inputValues[i] * v;
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpCpu::Div(const Tensor& input, float v, Tensor& output) const
    {
        input.CopyToHost();
        output.OverrideHost();

        auto& inputValues = input.GetValues();
        auto& outputValues = output.GetValues();

        for (uint32_t i = 0; i < inputValues.size(); ++i)
            outputValues[i] = inputValues[i] / v;
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpCpu::Div(const Tensor& t1, const Tensor& t2, Tensor& output) const
    {
        t1.CopyToHost();
        t2.CopyToHost();
        output.OverrideHost();

        for (uint32_t n = 0; n < max(t1.Batch(), t2.Batch()); ++n)
        {
            uint32_t t1N = n % t1.Batch();
            uint32_t t2N = n % t2.Batch();

            for (uint32_t d = 0; d < max(t1.Depth(), t2.Depth()); ++d)
            {
                uint32_t t1D = d % t1.Depth();
                uint32_t t2D = d % t2.Depth();

                for (uint32_t h = 0; h < max(t1.Height(), t2.Height()); ++h)
                {
                    uint32_t t1H = h % t1.Height();
                    uint32_t t2H = h % t2.Height();

                    for (uint32_t w = 0; w < max(t1.Width(), t2.Width()); ++w)
                    {
                        uint32_t t1W = w % t1.Width();
                        uint32_t t2W = w % t2.Width();

                        output(w, h, d, n) = t1(t1W, t1H, t1D, t1N) / t2(t2W, t2H, t2D, t2N);
                    }
                }
            }
        }
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpCpu::Add(const Tensor& input, float v, Tensor& output) const
    {
        input.CopyToHost();
        output.OverrideHost();

        auto& inputValues = input.GetValues();
        auto& outputValues = output.GetValues();

        for (uint32_t i = 0; i < inputValues.size(); ++i)
            outputValues[i] = inputValues[i] + v;
    }

    //////////////////////////////////////////////////////////////////////////
    template <int W, int H, int D, int N>
    void SumTemplate(const Tensor& input, Tensor& output)
    {
        auto& inputValues = input.GetValues();
        auto& outputValues = output.GetValues();
        auto& outputShape = output.GetShape();

        size_t i = 0;
        for (uint32_t n = 0; n < input.Batch(); ++n)
        for (uint32_t d = 0; d < input.Depth(); ++d)
        for (uint32_t h = 0; h < input.Height(); ++h)
        for (uint32_t w = 0; w < input.Width(); ++w, ++i)
            outputValues[outputShape.GetIndex(w * (1 - W), h * (1 - H), d * (1 - D), n * (1 - N))] += inputValues[i];
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpCpu::Sum(const Tensor& input, EAxis axis, Tensor& output) const
    {
        input.CopyToHost();
        output.OverrideHost();
        output.Zero();

        if (axis == EAxis::GlobalAxis)
            return SumTemplate<1, 1, 1, 1>(input, output);
        if (axis == EAxis::WidthAxis)
            return SumTemplate<1, 0, 0, 0>(input, output);
        else if (axis == EAxis::HeightAxis)
            return SumTemplate<0, 1, 0, 0>(input, output);
        else if (axis == EAxis::DepthAxis)
            return SumTemplate<0, 0, 1, 0>(input, output);
        else if (axis == EAxis::BatchAxis)
            return SumTemplate<0, 0, 0, 1>(input, output);
        else if (axis == EAxis::_01Axes)
            return SumTemplate<1, 1, 0, 0>(input, output);
        else if (axis == EAxis::_012Axes)
            return SumTemplate<1, 1, 1, 0>(input, output);
        else if (axis == EAxis::_013Axes)
            return SumTemplate<1, 1, 0, 1>(input, output);
        else if (axis == EAxis::_123Axes)
            return SumTemplate<0, 1, 1, 1>(input, output);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpCpu::Pow(const Tensor& input, float power, Tensor& output) const
    {
        input.CopyToHost();
        output.OverrideHost();

        auto& inputValues = input.GetValues();
        auto& outputValues = output.GetValues();

        for (uint32_t i = 0; i < inputValues.size(); ++i)
            outputValues[i] = ::pow(inputValues[i], power);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpCpu::PowGradient(const Tensor& input, float power, const Tensor& outputGradient, Tensor& inputGradient) const
    {
        input.CopyToHost();
        outputGradient.CopyToHost();
        inputGradient.OverrideHost();

        if (power == 2)
            outputGradient.Map([&](float g, float x) {return g * 2.f * x; }, input, inputGradient);
        else
            outputGradient.Map([&](float g, float x) {return g * power * ::pow(x, power - 1); }, input, inputGradient);
    }

    //////////////////////////////////////////////////////////////////////////
	void TensorOpCpu::Transpose(const Tensor& input, Tensor& output) const
	{
		input.CopyToHost();
        output.OverrideHost();

		for (uint32_t n = 0; n < input.Batch(); ++n)
		for (uint32_t d = 0; d < input.Depth(); ++d)
		for (uint32_t h = 0; h < input.Height(); ++h)
		for (uint32_t w = 0; w < input.Width(); ++w)
			output(h, w, d, n) = input(w, h, d, n);
	}

	//////////////////////////////////////////////////////////////////////////
	void TensorOpCpu::Map(const function<float(float)>& func, const Tensor& t, Tensor& output) const
	{
		t.CopyToHost();
        output.OverrideHost();

        auto& tValues = t.GetValues();
        auto& outputValues = output.GetValues();

		for (uint32_t i = 0; i < (uint32_t)tValues.size(); ++i)
			outputValues[i] = func(tValues[i]);
	}

	//////////////////////////////////////////////////////////////////////////
	void TensorOpCpu::Map(const function<float(float, float)>& func, const Tensor& t1, const Tensor& t2, Tensor& output) const
	{
        t1.CopyToHost();
        t2.CopyToHost();
        output.OverrideHost();

        for (uint32_t n = 0; n < max(t1.Batch(), t2.Batch()); ++n)
        {
            uint32_t t1N = n % t1.Batch();
            uint32_t t2N = n % t2.Batch();

            for (uint32_t d = 0; d < max(t1.Depth(), t2.Depth()); ++d)
            {
                uint32_t t1D = d % t1.Depth();
                uint32_t t2D = d % t2.Depth();

                for (uint32_t h = 0; h < max(t1.Height(), t2.Height()); ++h)
                {
                    uint32_t t1H = h % t1.Height();
                    uint32_t t2H = h % t2.Height();

                    for (uint32_t w = 0; w < max(t1.Width(), t2.Width()); ++w)
                    {
                        uint32_t t1W = w % t1.Width();
                        uint32_t t2W = w % t2.Width();

                        output(w, h, d, n) = func(t1(t1W, t1H, t1D, t1N), t2(t2W, t2H, t2D, t2N));
                    }
                }
            }
        }
	}

    //////////////////////////////////////////////////////////////////////////
    void TensorOpCpu::Sigmoid(const Tensor& input, Tensor& output) const
    {
        input.Map([&](float x) { return 1 / (1 + (float)exp(-x)); }, output);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpCpu::SigmoidGradient(const Tensor& output, const Tensor& outputGradient, Tensor& inputGradient) const
    {
        output.Map([&](float x, float x2) { return x * (1 - x) * x2; }, outputGradient, inputGradient);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpCpu::Tanh(const Tensor& input, Tensor& output) const
    {
        input.Map([&](float x) { return 2 / (1 + (float)exp(-2 * x)) - 1; }, output);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpCpu::TanhGradient(const Tensor& output, const Tensor& outputGradient, Tensor& inputGradient) const
    {
        output.Map([&](float x, float x2) { return (1 - x * x) * x2; }, outputGradient, inputGradient);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpCpu::ReLU(const Tensor& input, Tensor& output) const
    {
        input.Map([&](float x) { return max(0.f, x); }, output);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpCpu::ReLUGradient(const Tensor& output, const Tensor& outputGradient, Tensor& inputGradient) const
    {
        output.Map([&](float x, float x2) { return x > 0 ? x2 : 0; }, outputGradient, inputGradient);
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
	void TensorOpCpu::Softmax(const Tensor& input, Tensor& output) const
	{
		input.CopyToHost();
        output.OverrideHost();

		Tensor shifted = input.Sub(input.Max(EAxis::GlobalAxis)(0));
        Tensor exps = shifted.Map([&](float x) { return (float)exp(x); });

        auto& expsValues = exps.GetValues();
        auto& outputValues = output.GetValues();

        Tensor sum = exps.Sum(EAxis::_012Axes);
        sum.Reshape(Shape(sum.Batch()));

		for (uint32_t n = 0; n < input.Batch(); ++n)
		{
            for (uint32_t i = 0, idx = n * input.BatchLength(); i < input.BatchLength(); ++i, ++idx)
                outputValues[idx] = expsValues[idx] / sum(n);
		}
	}

	//////////////////////////////////////////////////////////////////////////
	void TensorOpCpu::SoftmaxGradient(const Tensor& output, const Tensor& outputGradient, Tensor& inputGradient) const
	{
		output.CopyToHost();
		outputGradient.CopyToHost();
        inputGradient.OverrideHost();
        inputGradient.Zero();

		Tensor outputReshaped = output.Reshaped(Shape(1, Shape::Auto, 1, output.Batch()));
		Tensor jacob = outputReshaped.DiagFlat().Sub(outputReshaped.Mul(outputReshaped.Transposed()));
        outputGradient.Mul(jacob, inputGradient);
	}

    //////////////////////////////////////////////////////////////////////////
    void TensorOpCpu::AdamStep(Tensor& parameter, const Tensor& gradient, Tensor& mGrad, Tensor& vGrad, float batchSize, float lr, float beta1, float beta2, float epsilon) const
    {
        parameter.CopyToHost();
        gradient.CopyToHost();
        mGrad.CopyToHost();
        vGrad.CopyToHost();

        float gradScale = 1.f / batchSize;
        float gradScale2 = gradScale * gradScale;
        
        // mGrad = beta1 * mGrad + (1 - beta1) * gradient
        mGrad.Add(beta1, (1 - beta1) * gradScale, gradient, mGrad);
        // vGrad = beta2 * vGrad + (1 - beta2) * sqr(gradient)
        vGrad.Map([&](float v, float g) { return v * beta2 + (1 - beta2) * gradScale2 * g * g ; }, gradient, vGrad);
        // parameter = parameter - mGrad / (sqrt(vGrad) + epsilon) * lr
        parameter.Sub(mGrad.Div(vGrad.Map([&](float x) { return (float)::sqrt(x) + epsilon; })).Mul(lr), parameter);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpCpu::SgdStep(Tensor& parameter, const Tensor& gradient, float batchSize, float lr) const
    {
        parameter.Add(1, -lr / batchSize, gradient, parameter);
    }

    //////////////////////////////////////////////////////////////////////////
	void TensorOpCpu::Conv2D(const Tensor& input, const Tensor& kernels, uint32_t stride, uint32_t paddingX, uint32_t paddingY, EDataFormat dataFormat, Tensor& output) const
	{
		input.CopyToHost();
		kernels.CopyToHost();
        output.OverrideHost();

        if (dataFormat == NCHW)
        {
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
        else
        {
            for (int n = 0; n < (int)input.Batch(); ++n)
		    for (int outD = 0; outD < (int)kernels.Batch(); ++outD)
		    for (int h = -(int)paddingY, outH = 0; outH < (int)output.Len(2); h += (int)stride, ++outH)
		    for (int w = -(int)paddingX, outW = 0; outW < (int)output.Len(1); w += (int)stride, ++outW)
		    {
			    float val = 0;

			    for (int kernelD = 0; kernelD < (int)kernels.Depth(); ++kernelD)
			    for (int kernelH = 0; kernelH < (int)kernels.Height(); ++kernelH)
			    for (int kernelW = 0; kernelW < (int)kernels.Width(); ++kernelW)
                    val += input.TryGet(0, kernelD, w + kernelW, h + kernelH, n) * kernels(kernelW, kernelH, kernelD, outD);

			    output(outD, outW, outH, n) = val;
		    }
        }
	}

	//////////////////////////////////////////////////////////////////////////
	void TensorOpCpu::Conv2DInputGradient(const Tensor& gradient, const Tensor& kernels, uint32_t stride, uint32_t paddingX, uint32_t paddingY, EDataFormat dataFormat, Tensor& inputGradient) const
	{
		gradient.CopyToHost();
		kernels.CopyToHost();
		inputGradient.OverrideHost();
        inputGradient.Zero();

        if (dataFormat == NCHW)
        {
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
                        if (inH >= 0 && inH < (int)inputGradient.Height() && inW >= 0 && inW < (int)inputGradient.Width())
                        {
                            for (int kernelD = 0; kernelD < (int)kernels.Depth(); ++kernelD)
                                inputGradient(inW, inH, kernelD, outN) += kernels.Get(kernelW, kernelH, kernelD, outD) * chainGradient;
                        }
                    }
                }
            }
        }
        else
        {
            for (int outN = 0; outN < (int)gradient.Batch(); ++outN)
            for (int outD = 0; outD < (int)gradient.Len(0); ++outD)
            for (int outH = 0, h = -(int)paddingY; outH < (int)gradient.Len(2); h += (int)stride, ++outH)
            for (int outW = 0, w = -(int)paddingX; outW < (int)gradient.Len(1); w += (int)stride, ++outW)
            {
                float chainGradient = gradient.Get(outD, outW, outH, outN);

                for (int kernelH = 0; kernelH < (int)kernels.Height(); ++kernelH)
                {
                    int inH = h + kernelH;
                    for (int kernelW = 0; kernelW < (int)kernels.Width(); ++kernelW)
                    {
                        int inW = w + kernelW;
                        if (inH >= 0 && inH < (int)inputGradient.Len(2) && inW >= 0 && inW < (int)inputGradient.Len(1))
                        {
                            for (int kernelD = 0; kernelD < (int)kernels.Depth(); ++kernelD)
                                inputGradient(kernelD, inW, inH, outN) += kernels.Get(kernelW, kernelH, kernelD, outD) * chainGradient;
                        }
                    }
                }
            }
        }
	}

	//////////////////////////////////////////////////////////////////////////
	void TensorOpCpu::Conv2DKernelsGradient(const Tensor& input, const Tensor& gradient, uint32_t stride, uint32_t paddingX, uint32_t paddingY, EDataFormat dataFormat, Tensor& kernelsGradient) const
	{
		input.CopyToHost();
		gradient.CopyToHost();
		kernelsGradient.OverrideHost();
        kernelsGradient.Zero();

        if (dataFormat == NCHW)
        {
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
        else
        {
            for (int outN = 0; outN < (int)gradient.Batch(); ++outN)
            for (int outD = 0; outD < (int)gradient.Len(0); ++outD)
            for (int outH = 0, h = -(int)paddingY; outH < (int)gradient.Len(2); h += (int)stride, ++outH)
            for (int outW = 0, w = -(int)paddingX; outW < (int)gradient.Len(1); w += (int)stride, ++outW)
            {
                float chainGradient = gradient.Get(outD, outW, outH, outN);

                for (int kernelH = 0; kernelH < (int)kernelsGradient.Height(); ++kernelH)
                {
                    int inH = h + kernelH;
                    for (int kernelW = 0; kernelW < (int)kernelsGradient.Width(); ++kernelW)
                    {
                        int inW = w + kernelW;
                        if (inH >= 0 && inH < (int)input.Len(2) && inW >= 0 && inW < (int)input.Len(1))
                        {
                            for (int kernelD = 0; kernelD < (int)kernelsGradient.Depth(); ++kernelD)
                                kernelsGradient(kernelW, kernelH, kernelD, outD) += input.Get(kernelD, inW, inH, outN) * chainGradient;
                        }
                    }
                }
            }
        }
	}

	//////////////////////////////////////////////////////////////////////////
	void TensorOpCpu::Pool2D(const Tensor& input, uint32_t filterSize, uint32_t stride, EPoolingMode type, uint32_t paddingX, uint32_t paddingY, EDataFormat dataFormat, Tensor& output) const
	{
		input.CopyToHost();
        output.OverrideHost();

        if (dataFormat == NCHW)
        {
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
        else
        {
            for (int outN = 0; outN < (int)input.Batch(); ++outN)
		    for (int outD = 0; outD < (int)input.Len(0); ++outD)
		    for (int outH = 0, h = -(int)paddingY; outH < (int)output.Len(2); h += (int)stride, ++outH)
		    for (int outW = 0, w = -(int)paddingX; outW < (int)output.Len(1); w += (int)stride, ++outW)
		    {
			    if (type == EPoolingMode::Max)
			    {
				    float value = -numeric_limits<float>().max();

				    for (int poolY = 0; poolY < (int)filterSize; ++poolY)
				    for (int poolX = 0; poolX < (int)filterSize; ++poolX)
					    value = max(value, input.TryGet(-numeric_limits<float>().max(), outD, w + poolX, h + poolY, outN));

				    output(outD, outW, outH, outN) = value;
			    }
			    else if (type == EPoolingMode::Avg)
			    {
				    float sum = 0;
				    for (int poolY = 0; poolY < (int)filterSize; ++poolY)
                    for (int poolX = 0; poolX < (int)filterSize; ++poolX)
                        sum += input.TryGet(0, outD, w + poolX, h + poolY, outN);

				    output(outD, outW, outH, outN) = sum / (filterSize * filterSize);
			    }
		    }
        }
	}

	//////////////////////////////////////////////////////////////////////////
    void TensorOpCpu::Pool2DGradient(const Tensor& output, const Tensor& input, const Tensor& outputGradient, uint32_t filterSize, uint32_t stride, EPoolingMode type, uint32_t paddingX, uint32_t paddingY, EDataFormat dataFormat, Tensor& inputGradient) const
	{
		output.CopyToHost();
		input.CopyToHost();
		outputGradient.CopyToHost();
		inputGradient.OverrideHost();
		inputGradient.Zero();

        if (dataFormat == NCHW)
        {
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
                    float filterElementsNum = (float)(filterSize * filterSize);

				    for (int poolH = 0; poolH < (int)filterSize; ++poolH)
				    for (int poolW = 0; poolW < (int)filterSize; ++poolW)
				    {
					    inputGradient.TrySet(inputGradient.TryGet(-numeric_limits<float>().max(), w + poolW, h + poolH, outD, outN) + outputGradient(outW, outH, outD, outN) / filterElementsNum, w + poolW, h + poolH, outD, outN);
				    }
			    }
		    }
        }
        else
        {
            for (int outN = 0; outN < (int)output.Batch(); ++outN)
		    for (int outD = 0; outD < (int)output.Len(0); ++outD)
		    for (int outH = 0, h = -(int)paddingY; outH < (int)output.Len(2); ++outH, h += (int)stride)
		    for (int outW = 0, w = -(int)paddingX; outW < (int)output.Len(1); ++outW, w += (int)stride)
		    {
			    if (type == EPoolingMode::Max)
			    {
                    bool maxFound = false;
                    for (int poolH = 0; poolH < (int)filterSize; ++poolH)
                    {
                        for (int poolW = 0; poolW < (int)filterSize; ++poolW)
                        {
                            float value = input.TryGet(-numeric_limits<float>().max(), w + poolW, h + poolH, outD, outN);
                            if (value == output(outD, outW, outH, outN))
                            {
                                inputGradient.TrySet(inputGradient.TryGet(-numeric_limits<float>().max(), outD, w + poolW, h + poolH, outN) + outputGradient(outD, outW, outH, outN), outD, w + poolW, h + poolH, outN);
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
					    inputGradient.TrySet(inputGradient.TryGet(-numeric_limits<float>().max(), outD, w + poolW, h + poolH, outN) + outputGradient(outD, outW, outH, outN) / filterElementsNum, outD, w + poolW, h + poolH, outN);
				    }
			    }
		    }
        }
	}

    //////////////////////////////////////////////////////////////////////////
    void TensorOpCpu::UpSample2D(const Tensor& input, uint32_t scaleFactor, Tensor& output) const
    {
        output.CopyToHost();
        output.OverrideHost();
        
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
        outputGradient.CopyToHost();
        inputGradient.OverrideHost();
        inputGradient.Zero();

        for (uint32_t n = 0; n < outputGradient.Batch(); ++n)
        for (uint32_t d = 0; d < outputGradient.Depth(); ++d)
        for (uint32_t h = 0; h < outputGradient.Height(); ++h)
        for (uint32_t w = 0; w < outputGradient.Width(); ++w)
            inputGradient(w / scaleFactor, h / scaleFactor, d, n) += outputGradient(w, h, d, n);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpCpu::BatchNormalization(const Tensor& input, EBatchNormMode mode, const Tensor& gamma, const Tensor& beta, float epsilon, const Tensor* runningMean, const Tensor* runningVar, Tensor& output) const
    {
        input.CopyToHost();
        gamma.CopyToHost();
        beta.CopyToHost();
        if (runningMean)
            runningMean->CopyToHost();
        if (runningVar)
            runningVar->CopyToHost();
        output.OverrideHost();

        Tensor xNorm;

        if (runningMean && runningVar)
        {
            Tensor xMu = input - *runningMean;
            xNorm = xMu * (1.f / sqrt(*runningVar + epsilon));
        }
        else
        {
            NEURO_ASSERT(mode == Instance, "Running mean and variance can be missing only for Instance normalization.");
            Tensor xMu = input - mean(input, _01Axes);
            Tensor xVar = mean(sqr(xMu), _01Axes);
            xNorm = xMu * (1.f / sqrt(xVar + epsilon));
        }

        xNorm.MulElem(gamma).Add(beta, output);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpCpu::BatchNormalizationTrain(const Tensor& input, EBatchNormMode mode, const Tensor& gamma, const Tensor& beta, float momentum, float epsilon, Tensor* runningMean, Tensor* runningVar, Tensor& saveMean, Tensor& saveInvVariance, Tensor& output) const
    {
        input.CopyToHost();
        gamma.CopyToHost();
        beta.CopyToHost();
        if (runningMean)
            runningMean->CopyToHost();
        if (runningVar)
            runningVar->CopyToHost();
        saveMean.CopyToHost();
        saveInvVariance.CopyToHost();
        output.OverrideHost();

        EAxis axis;
        float m;
        
        if (mode == PerActivation)
        {
            axis = BatchAxis; // mean is of shape WxHxDx1
            m = (float)input.Batch();
        }
        else if (mode == Spatial)
        {
            axis = _013Axes; // mean is of shape 1x1xDx1 // for NHWC format it must be _123Axes
            m = (float)(input.Width() * input.Height() * input.Batch());
        }
        else if (mode == Instance)
        {
            axis = _01Axes; // output is (1x1xDxN)
            m = (float)(input.Width() * input.Height());
        }

        input.Mean(axis, saveMean);
        Tensor xMu = input - saveMean;
        Tensor var = mean(sqr(xMu), axis);
        Tensor varSqrt = sqrt(var + epsilon);
        varSqrt.Map([](float x) { return 1.f / x; }, saveInvVariance);
        Tensor xNorm = xMu * saveInvVariance;
        xNorm.MulElem(gamma).Add(beta, output);

        if (runningMean)
            runningMean->Add(1 - momentum, momentum, saveMean, *runningMean);

        if (runningVar)
        {
            Tensor tempVar = var * m / (m - 1); // according to the original BN paper
            runningVar->Add(1 - momentum, momentum, tempVar, *runningVar);
        }
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpCpu::BatchNormalizationGradient(const Tensor& input, EBatchNormMode mode, const Tensor& gamma, float epsilon, const Tensor& outputGradient, const Tensor& savedMean, const Tensor& savedInvVariance, Tensor& gammaGradient, Tensor& betaGradient, bool trainable, Tensor& inputGradient) const
    {
        input.CopyToHost();
        gamma.CopyToHost();
        outputGradient.CopyToHost();
        savedMean.CopyToHost();
        savedInvVariance.CopyToHost();
        gammaGradient.OverrideHost();
        gammaGradient.Zero();
        betaGradient.OverrideHost();
        betaGradient.Zero();
        inputGradient.OverrideHost();
        inputGradient.Zero();

        EAxis axis;
        float m;

        if (mode == PerActivation)
        {
            axis = BatchAxis; // mean is of shape WxHxDx1
            m = (float)input.Batch();
        }
        else if (mode == Spatial)
        {
            axis = _013Axes; // mean is of shape 1x1xDx1 // for NHWC format it must be _123Axes
            m = (float)(input.Width() * input.Height() * input.Batch());
        }
        else if (mode == Instance)
        {
            axis = _01Axes; // output is (1x1xDxN)
            m = (float)(input.Width() * input.Height());
        }

        Tensor xMu = input - savedMean;
        Tensor xNorm = xMu * savedInvVariance;
        Tensor dxNorm = outputGradient * gamma;
        Tensor dVar = sum(dxNorm * xMu, axis) * -.5f * pow(savedInvVariance, 3);
        Tensor dMu = sum(dxNorm * -savedInvVariance, axis) + dVar * mean(xMu * -2.f, axis);

        inputGradient = (dxNorm * savedInvVariance) + (dVar * xMu * 2.f / m) + (dMu / m);
        gammaGradient = sum(outputGradient * xNorm, axis);
        betaGradient = sum(outputGradient, axis);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpCpu::Dropout(const Tensor& input, float prob, Tensor& saveMask, Tensor& output)
    {
        input.CopyToHost();
        saveMask.OverrideHost();
        output.OverrideHost();

        saveMask.FillWithFunc([&]() { return (GlobalRng().NextFloat() < prob ? 0.f : 1.f) / prob; });
        input.MulElem(saveMask, output);
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorOpCpu::DropoutGradient(const Tensor& outputGradient, const Tensor& savedMask, Tensor& inputGradient)
    {
        outputGradient.CopyToHost();
        savedMask.CopyToHost();
        inputGradient.OverrideHost();
        inputGradient.Zero();

        outputGradient.MulElem(savedMask, inputGradient);
    }
}
