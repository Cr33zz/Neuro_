#include <algorithm>

#include "Tensors/Tensor.h"
#include "Tensors/Cuda/CudaDeviceVariable.h"
#include "Tensors/TensorOpCpu.h"
#include "Tensors/TensorOpMultiCpu.h"
#include "Tensors/TensorOpGpu.h"
#include "Random.h"
#include "Tools.h"

namespace Neuro
{
    using namespace std;

	TensorOpCpu* Tensor::g_OpCpu = new TensorOpCpu();
    TensorOpCpu* Tensor::g_OpMultiCpu = nullptr; //new TensorOpMultiCpu();
    TensorOpCpu* Tensor::g_OpGpu = nullptr; //new TensorOpGpu();
	TensorOpCpu* Tensor::g_DefaultOpCpu = nullptr;

	//////////////////////////////////////////////////////////////////////////
	Tensor::Tensor()
	{
		OverrideHost();

		if (g_DefaultOpCpu == nullptr)
			g_DefaultOpCpu = g_OpCpu;

		m_Op = g_DefaultOpCpu;
	}

	//////////////////////////////////////////////////////////////////////////
	Tensor::Tensor(const Shape& shape)
		: Tensor()
	{
		m_Shape = shape;
		m_Values.resize(shape.Length);
	}

	//////////////////////////////////////////////////////////////////////////
	Tensor::Tensor(const Tensor& t)
		: Tensor()
	{
		t.CopyToHost();
		m_Shape = t.GetShape();
		m_Values = t.m_Values;
	}

	//////////////////////////////////////////////////////////////////////////
	Tensor::Tensor(const vector<float>& values)
		: Tensor()
	{
		m_Shape = Shape((int)values.size());
		m_Values = values;
	}

	//////////////////////////////////////////////////////////////////////////
	Tensor::Tensor(const vector<float>& values, const Shape& shape)
		: Tensor()
	{
		assert(values.size() == shape.Length);// && string("Invalid array size ") + to_string(values.size()) + ". Expected " + to_string(shape.Length) + ".");
		m_Shape = shape;
		m_Values = values;
	}

	//////////////////////////////////////////////////////////////////////////
	Tensor::Tensor(string bmpFile, bool grayScale)
	{
		/*using (var bmp = new Bitmap(bmpFile))
		{
			Shape = Shape(bmp.Width, bmp.Height, grayScale ? 1 : 3);
			Values = new float[Shape.Length];

			for (int h = 0; h < bmp.Height; ++h)
			{
				for (int w = 0; w < bmp.Width; ++w)
				{
					Color c = bmp.GetPixel(w, h);
					if (grayScale)
						Set((c.R * 0.3f + c.G * 0.59f + c.B * 0.11f) / 255.0f, w, h);
					else
					{
						Set(c.R / 255.0f, w, h, 0);
						Set(c.G / 255.0f, w, h, 1);
						Set(c.B / 255.0f, w, h, 2);
					}
				}
			}
		}*/
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::SetDefaultOpMode(EOpMode mode)
	{
		g_DefaultOpCpu = GetOpMode(mode);
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::SetOpMode(EOpMode mode)
	{
		m_Op = GetOpMode(mode);
	}

	//////////////////////////////////////////////////////////////////////////
	std::vector<float>& Tensor::GetValues()
	{
		CopyToHost();
		return m_Values;
	}

    //////////////////////////////////////////////////////////////////////////
    const std::vector<float>& Tensor::GetValues() const
    {
        CopyToHost();
        return m_Values;
    }

	//////////////////////////////////////////////////////////////////////////
	Tensor& Tensor::FillWithRand(int seed, float min, float max)
	{
		OverrideHost();

		auto fillUp = [&](const Random& rng)
		{
			for (int i = 0; i < m_Values.size(); ++i)
				m_Values[i] = min + (max - min) * rng.NextFloat();
		};

		fillUp(seed > 0 ? Random(seed) : Rng);

		return *this;
	}

	//////////////////////////////////////////////////////////////////////////
	Tensor& Tensor::FillWithRange(float start, float increment)
	{
		OverrideHost();
		for (int i = 0; i < m_Values.size(); ++i)
			m_Values[i] = start + i * increment;
		return *this;
	}

	//////////////////////////////////////////////////////////////////////////
	Tensor& Tensor::FillWithValue(float value)
	{
		OverrideHost();
		for (int i = 0; i < m_Values.size(); ++i)
			m_Values[i] = value;
		return *this;
	}

    //////////////////////////////////////////////////////////////////////////
    Tensor& Tensor::FillWithFunc(const function<float()>& func)
    {
        OverrideHost();
        for (int i = 0; i < m_Values.size(); ++i)
            m_Values[i] = func();
        return *this;
    }

    //////////////////////////////////////////////////////////////////////////
	void Tensor::Zero()
	{
		OverrideHost();
		fill(m_Values.begin(), m_Values.end(), 0.f);
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::Mul(bool transposeT, const Tensor& t, Tensor& result) const
	{
		assert((!transposeT && Width() == t.Height()) || (transposeT && Width() == t.Width()));
		assert(t.Depth() == Depth());

		m_Op->Mul(false, transposeT, *this, t, result);
	}

	//////////////////////////////////////////////////////////////////////////
	Tensor Tensor::Mul(bool transposeT, const Tensor& t) const
	{
		Tensor result(Shape(transposeT ? t.m_Shape.Height() : t.m_Shape.Width(), Height(), Depth(), Batch()));
		Mul(transposeT, t, result);
		return result;
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::Mul(const Tensor& t, Tensor& result) const
	{
		Mul(false, t, result);
	}

	//////////////////////////////////////////////////////////////////////////
	Tensor Tensor::Mul(const Tensor& t) const
	{
		return Mul(false, t);
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::MulElem(const Tensor& t, Tensor& result) const
	{
		assert(SameDimensionsExceptBatches(t));

		m_Op->MulElem(*this, t, result);
	}


	//////////////////////////////////////////////////////////////////////////
	Tensor Tensor::MulElem(const Tensor& t) const
	{
		Tensor result(m_Shape);
		MulElem(t, result);
		return result;
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::Mul(float v, Tensor& result) const
	{
		CopyToHost();
		result.OverrideHost();

		for (int i = 0; i < m_Values.size(); ++i)
			result.m_Values[i] = m_Values[i] * v;
	}

	//////////////////////////////////////////////////////////////////////////
	Tensor Tensor::Mul(float v) const
	{
		Tensor result(m_Shape);
		Mul(v, result);
		return result;
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::Div(const Tensor& t, Tensor& result) const
	{
		CopyToHost();
		result.OverrideHost();

		assert(SameDimensionsExceptBatches(t));
		assert(t.Batch() == result.Batch());

		for (int i = 0; i < m_Values.size(); ++i)
			result.m_Values[i] = m_Values[i] / t.m_Values[i];
	}


	//////////////////////////////////////////////////////////////////////////
	Tensor Tensor::Div(const Tensor& t) const
	{
		Tensor result(m_Shape);
		Div(t, result);
		return result;
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::Div(float v, Tensor& result) const
	{
		CopyToHost();
		result.OverrideHost();

		for (int i = 0; i < m_Values.size(); ++i)
			result.m_Values[i] = m_Values[i] / v;
	}

	//////////////////////////////////////////////////////////////////////////
	Tensor Tensor::Div(float v) const
	{
		Tensor result(m_Shape);
		Div(v, result);
		return result;
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::Add(float alpha, float beta, const Tensor& t, Tensor& result) const
	{
		assert(SameDimensionsExceptBatches(t));
		assert(t.Batch() == result.Batch() || t.Batch() == 1);

		m_Op->Add(alpha, *this, beta, t, result);
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::Add(const Tensor& t, Tensor& result) const
	{
		Add(1, 1, t, result);
	}

	//////////////////////////////////////////////////////////////////////////
	Tensor Tensor::Add(const Tensor& t) const
	{
		Tensor result(m_Shape);
		Add(t, result);
		return result;
	}

	//////////////////////////////////////////////////////////////////////////
	Tensor Tensor::Add(float alpha, float beta, const Tensor& t) const
	{
		Tensor result(m_Shape);
		Add(alpha, beta, t, result);
		return result;
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::Add(float v, Tensor& result) const
	{
		CopyToHost();
		for (int i = 0; i < m_Values.size(); ++i)
			result.m_Values[i] = m_Values[i] + v;
	}

	//////////////////////////////////////////////////////////////////////////
	Tensor Tensor::Add(float v) const
	{
		Tensor result(m_Shape);
		Add(v, result);
		return result;
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::Sub(const Tensor& t, Tensor& result) const
	{
		assert(SameDimensionsExceptBatches(t));
		assert(t.Batch() == result.Batch() || t.Batch() == 1);

		m_Op->Sub(*this, t, result);
	}

	//////////////////////////////////////////////////////////////////////////
	Tensor Tensor::Sub(const Tensor& t) const
	{
		Tensor result(m_Shape);
		Sub(t, result);
		return result;
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::Sub(float v, Tensor& result) const
	{
		CopyToHost();
		for (int i = 0; i < m_Values.size(); ++i)
			result.m_Values[i] = m_Values[i] - v;
	}

	//////////////////////////////////////////////////////////////////////////
	Tensor Tensor::Sub(float v) const
	{
		Tensor result(m_Shape);
		Sub(v, result);
		return result;
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::Negated(Tensor& result) const
	{
		for (int i = 0; i < m_Values.size(); ++i)
			result.m_Values[i] = -m_Values[i];
	}

	//////////////////////////////////////////////////////////////////////////
	Tensor Tensor::Negated() const
	{
		Tensor result(m_Shape);
		Negated(result);
		return result;
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::Clipped(float min, float max, Tensor& result) const
	{
		Map([&](float x) { return Clip(x, min, max); }, result);
	}

	//////////////////////////////////////////////////////////////////////////
	Tensor Tensor::Clipped(float min, float max) const
	{
		Tensor result(m_Shape);
		Clipped(min, max, result);
		return result;
	}

	//////////////////////////////////////////////////////////////////////////
	Tensor Tensor::DiagFlat() const
	{
		Tensor result(Shape(BatchLength(), BatchLength(), 1, Batch()));

		int batchLen = BatchLength();

		for (int b = 0; b < Batch(); ++b)
			for (int i = 0; i < batchLen; ++i)
				result(i, i, 0, b) = m_Values[b * batchLen + i];

		return result;
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::Map(const function<float(float)>& func, Tensor& result) const
	{
		m_Op->Map(func, *this, result);
	}

	//////////////////////////////////////////////////////////////////////////
	Tensor Tensor::Map(const function<float(float)>& func) const
	{
		Tensor result(m_Shape);
		Map(func, result);
		return result;
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::Map(const function<float(float, float)>& func, const Tensor& other, Tensor& result) const
	{
		m_Op->Map(func, *this, other, result);
	}

	//////////////////////////////////////////////////////////////////////////
	Tensor Tensor::Map(const function<float(float, float)>& func, const Tensor& other) const
	{
		Tensor result(m_Shape);
		Map(func, other, result);
		return result;
	}

	//////////////////////////////////////////////////////////////////////////
	Tensor Tensor::SumBatches() const
	{
		Tensor result(Shape(m_Shape.Width(), m_Shape.Height(), m_Shape.Depth(), 1));
		m_Op->SumBatches(*this, result);
		return result;
	}

	//////////////////////////////////////////////////////////////////////////
	float Tensor::Sum(int batch) const
	{
		CopyToHost();
		int batchLen = BatchLength();

		if (batch < 0)
		{
			batchLen = Length();
			batch = 0;
		}

		float sum = 0;

		for (int i = 0, idx = batch * batchLen; i < batchLen; ++i, ++idx)
			sum += m_Values[idx];

		return sum;
	}

	//////////////////////////////////////////////////////////////////////////
	Tensor Tensor::SumPerBatch() const
	{
		CopyToHost();
		Tensor result(Shape(1, 1, 1, m_Shape.Batch()));

		for (int n = 0; n < Batch(); ++n)
			result.m_Values[n] = Sum(n);

		return result;
	}

	//////////////////////////////////////////////////////////////////////////
	Tensor Tensor::AvgBatches() const
	{
		CopyToHost();
		Tensor result = SumBatches();

		int batchLen = BatchLength();

		for (int n = 0; n < batchLen; ++n)
			result.m_Values[n] /= Batch();

		return result;
	}

	//////////////////////////////////////////////////////////////////////////
	float Tensor::Avg(int batch) const
	{
		return Sum(batch) / (batch < 0 ? Length() : BatchLength());
	}

	//////////////////////////////////////////////////////////////////////////
	Tensor Tensor::AvgPerBatch() const
	{
		Tensor result = SumPerBatch();

		int batchLen = BatchLength();

		for (int n = 0; n < Batch(); ++n)
			result.m_Values[n] /= batchLen;

		return result;
	}

	//////////////////////////////////////////////////////////////////////////
	float Tensor::Max(int batch) const
	{
		int maxIndex;
		return GetMaxData(batch, maxIndex);
	}

	//////////////////////////////////////////////////////////////////////////
    Tensor Tensor::MaxPerBatch() const
	{
		Tensor result(Shape(1, 1, 1, m_Shape.Batch()));
		for (int n = 0; n < Batch(); ++n)
			result(0, 0, 0, n) = Max(n);
		return result;
	}

    //////////////////////////////////////////////////////////////////////////
	Tensor Tensor::MergeIntoBatch(const vector<Tensor>& tensors)
	{
		/*if (tensors.Count == 0)
			throw new Exception("List cannot be empty.");*/

		Tensor output(Shape(tensors[0].Width(), tensors[0].Height(), tensors[0].Depth(), (int)tensors.size()));

		for (int n = 0; n < tensors.size(); ++n)
		{
			const Tensor& t = tensors[n];
			t.CopyToHost();
			copy(t.m_Values.begin(), t.m_Values.end(), output.m_Values.begin() + t.Length() * n);
		}

		return output;
	}

	//////////////////////////////////////////////////////////////////////////
	Tensor Tensor::MergeIntoDepth(const vector<Tensor>& tensors, int forcedDepth)
	{
		/*if (tensors.Count == 0)
			throw new Exception("List cannot be empty.");*/

		Tensor output(Shape(tensors[0].Width(), tensors[0].Height(), max((int)tensors.size(), forcedDepth)));

		const Tensor& t = tensors[0];
		t.CopyToHost();

		int t0_copies = forcedDepth > 0 ? forcedDepth - (int)tensors.size() : 0;

		for (int n = 0; n < t0_copies; ++n)
		{
			copy(t.m_Values.begin(), t.m_Values.end(), output.m_Values.begin() + t.Length() * n);
		}

		for (int n = t0_copies; n < output.Depth(); ++n)
		{
			const Tensor& t = tensors[n - t0_copies];
			t.CopyToHost();
			copy(t.m_Values.begin(), t.m_Values.end(), output.m_Values.begin() + t.Length() * n);
		}

		return output;
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::Concat(const tensor_ptr_vec_t& inputs, Tensor& result)
	{
		for (int b = 0; b < result.Batch(); ++b)
		{
			int elementsCopied = 0;
			for (int i = 0; i < inputs.size(); ++i)
			{
				inputs[i]->CopyToHost();
				copy(inputs[i]->m_Values.begin() + b * inputs[i]->BatchLength(), inputs[i]->m_Values.begin() + (b + 1) * inputs[i]->BatchLength(), result.m_Values.begin() + b * result.BatchLength() + elementsCopied);
				elementsCopied += inputs[i]->BatchLength();
			}
		}
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::Split(vector<Tensor>& outputs) const
	{
		CopyToHost();
		for (int b = 0; b < Batch(); ++b)
		{
			int elementsCopied = 0;
			for (int i = 0; i < outputs.size(); ++i)
			{
				outputs[i].CopyToHost();
				copy(m_Values.begin() + b * BatchLength() + elementsCopied, 
                     m_Values.begin() + b * BatchLength() + elementsCopied + outputs[i].BatchLength(),
                     outputs[i].m_Values.begin() + b * outputs[i].BatchLength());
				elementsCopied += outputs[i].BatchLength();
			}
		}
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::MergeMin(const tensor_ptr_vec_t& inputs, Tensor& result)
	{
		inputs[0]->CopyTo(result);
		for (int i = 1; i < inputs.size(); ++i)
			for (int j = 0; j < result.Length(); ++j)
				result.m_Values[j] = result.m_Values[j] > inputs[i]->m_Values[j] ? inputs[i]->m_Values[j] : result.m_Values[j];
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::MergeMax(const tensor_ptr_vec_t& inputs, Tensor& result)
	{
		inputs[0]->CopyTo(result);
		for (int i = 1; i < inputs.size(); ++i)
			for (int j = 0; j < result.Length(); ++j)
				result.m_Values[j] = result.m_Values[j] < inputs[i]->m_Values[j] ? inputs[i]->m_Values[j] : result.m_Values[j];
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::MergeSum(const tensor_ptr_vec_t& inputs, Tensor& result)
	{
		result.Zero();
		for (int i = 0; i < inputs.size(); ++i)
			for (int j = 0; j < result.Length(); ++j)
				result.m_Values[j] += inputs[i]->m_Values[j];
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::MergeAvg(const tensor_ptr_vec_t& inputs, Tensor& result)
	{
		MergeSum(inputs, result);
		result.Div((float)inputs.size(), result);
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::MergeMinMaxGradient(const Tensor& output, const tensor_ptr_vec_t& inputs, const Tensor& outputGradient, vector<Tensor>& results)
	{
		for (int i = 0; i < inputs.size(); ++i)
		{
			results[i].Zero();
			for (int j = 0; j < output.Length(); ++j)
				results[i].m_Values[j] = inputs[i]->m_Values[j] == output.m_Values[j] ? outputGradient.m_Values[j] : 0;
		}
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::MergeSumGradient(const Tensor& output, const tensor_ptr_vec_t& inputs, const Tensor& outputGradient, vector<Tensor>& results)
	{
		for (int i = 0; i < inputs.size(); ++i)
			outputGradient.CopyTo(results[i]);
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::MergeAvgGradient(const Tensor& output, const tensor_ptr_vec_t& inputs, const Tensor& outputGradient, vector<Tensor>& results)
	{
		MergeSumGradient(output, inputs, outputGradient, results);
		for (int i = 0; i < results.size(); ++i)
			results[i].Div((float)results.size(), results[i]);
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::Normalized(Tensor& result) const
	{
		float sum = Sum();
        Map([&](float x) { return x / sum; }, result);
	}

	//////////////////////////////////////////////////////////////////////////
	int Tensor::ArgMax(int batch) const
	{
		int maxIndex;
		GetMaxData(batch, maxIndex);
		return maxIndex;
	}

	//////////////////////////////////////////////////////////////////////////
	Tensor Tensor::ArgMaxPerBatch() const
	{
		Tensor result(Shape(1, 1, 1, m_Shape.Batch()));
		for (int n = 0; n < Batch(); ++n)
			result(0, 0, 0, n) = (float)ArgMax(n);
		return result;
	}

	//////////////////////////////////////////////////////////////////////////
	Tensor Tensor::Transposed() const
	{
		Tensor result(Shape(Height(), Width(), Depth(), Batch()));
		Transpose(result);
		return result;
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::Transpose(Tensor& result) const
	{
		m_Op->Transpose(*this, result);
	}

	//////////////////////////////////////////////////////////////////////////
	Tensor Tensor::Reshaped(Shape shape) const
	{
		return Tensor(m_Values, m_Shape.Reshaped(shape.Width(), shape.Height(), shape.Depth(), shape.Batch()));
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::Reshape(Shape shape)
	{
		m_Shape = m_Shape.Reshaped(shape.Width(), shape.Height(), shape.Depth(), shape.Batch());
	}

	//////////////////////////////////////////////////////////////////////////
	Tensor Tensor::Resized(int width, int height, int depth) const
	{
		int newBatchLength = width * height * depth;
		Tensor result(Shape(width, height, depth, m_Shape.Batch()));
		for (int n = 0; n < Batch(); ++n)
			for (int i = 0, idx = n * newBatchLength; i < newBatchLength; ++i, ++idx)
				result.m_Values[idx] = m_Values[n * BatchLength() + i % BatchLength()];
		return result;
	}

	//////////////////////////////////////////////////////////////////////////
	Tensor Tensor::FlattenHoriz() const
	{
		return Reshaped(m_Shape.Reshaped(Shape::Auto, 1, 1, m_Shape.Batch()));
	}

	//////////////////////////////////////////////////////////////////////////
	Tensor Tensor::FlattenVert() const
	{
		return Reshaped(m_Shape.Reshaped(1, Shape::Auto, 1, m_Shape.Batch()));
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::Rotated180(Tensor& result) const
	{
		assert(SameDimensionsExceptBatches(result));

		for (int n = 0; n < Batch(); ++n)
			for (int d = 0; d < Depth(); ++d)
				for (int h = Height() - 1; h >= 0; --h)
					for (int w = Width() - 1; w >= 0; --w)
						result.Set(Get(Width() - w - 1, Height() - h - 1, d, n), w, h, d, n);
	}

	//////////////////////////////////////////////////////////////////////////
	Tensor Tensor::Rotated180() const
	{
		Tensor result(m_Shape);
		Rotated180(result);
		return result;
	}

	//////////////////////////////////////////////////////////////////////////
    void Tensor::Conv2D(const Tensor& kernels, int stride, EPaddingType padding, Tensor& result) const
	{
		assert(Depth() == kernels.Depth());

		m_Op->Conv2D(*this, kernels, stride, padding, result);
	}

	//////////////////////////////////////////////////////////////////////////
	Tensor Tensor::Conv2D(const Tensor& kernels, int stride, EPaddingType padding) const
	{
		int outputWidth = 0, outputHeight = 0, paddingX = 0, paddingY = 0;
		GetPaddingParams(padding, Width(), Height(), kernels.Width(), kernels.Height(), stride, outputHeight, outputWidth, paddingX, paddingY);

		Tensor result(Shape(outputWidth, outputHeight, kernels.Batch(), Batch()));
		Conv2D(kernels, stride, padding, result);
		return result;
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::Conv2DInputsGradient(const Tensor& gradient, const Tensor& kernels, int stride, EPaddingType padding, Tensor& inputsGradient) const
	{
		inputsGradient.Zero();
		m_Op->Conv2DInputGradient(gradient, kernels, stride, padding, inputsGradient);
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::Conv2DKernelsGradient(const Tensor& input, const Tensor& gradient, int stride, EPaddingType padding, Tensor& kernelsGradient) const
	{
		kernelsGradient.Zero();
		m_Op->Conv2DKernelsGradient(input, gradient, stride, padding, kernelsGradient);
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::Conv2DGradient_old(const Tensor& input, const Tensor& kernels, const Tensor& gradient, int stride, EPaddingType padding, Tensor& inputGradient, Tensor& kernelsGradient) const
	{
		inputGradient.Zero();
		kernelsGradient.Zero();
		int outputWidth = 0, outputHeight = 0, paddingX = 0, paddingY = 0;
		GetPaddingParams(padding, input.Width(), input.Height(), kernels.Width(), kernels.Height(), stride, outputHeight, outputWidth, paddingX, paddingY);
		m_Op->Conv2DGradient_old(input, kernels, gradient, stride, paddingX, paddingY, inputGradient, kernelsGradient);
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::Pool(int filterSize, int stride, EPoolType type, EPaddingType padding, Tensor& result) const
	{
		int outWidth = 0, outHeight = 0, paddingX = 0, paddingY = 0;
		GetPaddingParams(padding, Width(), Height(), filterSize, filterSize, stride, outHeight, outWidth, paddingX, paddingY);

		assert(result.Width() == outWidth);
		assert(result.Height() == outHeight);
		assert(result.Batch() == Batch());

		m_Op->Pool(*this, filterSize, stride, type, paddingX, paddingY, result);
	}

	//////////////////////////////////////////////////////////////////////////
	Tensor Tensor::Pool(int filterSize, int stride, EPoolType type, EPaddingType padding) const
	{
		int outWidth = 0, outHeight = 0, paddingX = 0, paddingY = 0;
		GetPaddingParams(padding, Width(), Height(), filterSize, filterSize, stride, outHeight, outWidth, paddingX, paddingY);

		Tensor result(Shape(outWidth, outHeight, Depth(), Batch()));
		Pool(filterSize, stride, type, padding, result);

		return result;
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::PoolGradient(const Tensor& output, const Tensor& input, const Tensor& outputGradient, int filterSize, int stride, EPoolType type, EPaddingType padding, Tensor& result) const
	{
		assert(output.SameDimensionsExceptBatches(outputGradient));

		int outWidth = 0, outHeight = 0, paddingX = 0, paddingY = 0;
		GetPaddingParams(padding, result.Width(), result.Height(), filterSize, filterSize, stride, outHeight, outWidth, paddingX, paddingY);

		m_Op->PoolGradient(output, input, outputGradient, filterSize, stride, type, paddingX, paddingY, result);
	}

	//////////////////////////////////////////////////////////////////////////
	std::string Tensor::ToString() const
	{
		string s = "";
		/*for (int n = 0; n < Batch(); ++n)
		{
			if (Batch() > 1)
				s += "{\n  ";

			for (int d = 0; d < Depth(); ++d)
			{
				if (Depth() > 1)
					s += "{\n    ";

				for (int h = 0; h < Height(); ++h)
				{
					s += "{ ";
					for (int w = 0; w < Width(); ++w)
					{
						s += Get(w, h, d, n) + (w == Width() - 1 ? "" : ", ");
					}
					s += " }" + (h == Height() - 1 ? "\n  " : ",\n    ");
				}

				if (Depth() > 1)
					s += "}" + (d == Depth() - 1 ? "\n" : ",\n  ");
			}

			if (Batch() > 1)
				s += "}" + (n < Batch() - 1 ? "\n" : "");
		}*/

		return s;
	}

	//////////////////////////////////////////////////////////////////////////
	bool Tensor::SameDimensionsExceptBatches(const Tensor& t) const
	{
		return Width() == t.Width() && Height() == t.Height() && Depth() == t.Depth();
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::GetPaddingParams(EPaddingType type, int width, int height, int kernelWidth, int kernelHeight, int stride, int& outHeight, int& outWidth, int& paddingX, int& paddingY)
	{
		if (type == EPaddingType::Valid)
		{
			outWidth = (int)floor((width - kernelWidth) / (float)stride + 1);
			outHeight = (int)floor((height - kernelHeight) / (float)stride + 1);
			paddingX = 0;
			paddingY = 0;
		}
		else if (type == EPaddingType::Same)
		{
			outWidth = width / stride;
			outHeight = height / stride;
			paddingX = (int)floor((float)kernelWidth / 2);
			paddingY = (int)floor((float)kernelHeight / 2);
		}
		else //if (type == ConvType.Full)
		{
			outWidth = (width + (kernelWidth - 1)) / stride;
			outHeight = (height + (kernelHeight - 1)) / stride;
			paddingX = kernelWidth - 1;
			paddingY = kernelHeight - 1;
		}
	}

	//////////////////////////////////////////////////////////////////////////
	float Tensor::GetFlat(int i)
	{
		CopyToHost();
		return m_Values[i];
	}

	//////////////////////////////////////////////////////////////////////////
	float& Tensor::Get(int w, int h, int d, int n)
	{
		CopyToHost();
		return m_Values[m_Shape.GetIndex(w, h, d, n)];
	}

    //////////////////////////////////////////////////////////////////////////
    float Tensor::Get(int w, int h, int d, int n) const
    {
        CopyToHost();
        return m_Values[m_Shape.GetIndex(w, h, d, n)];
    }

	//////////////////////////////////////////////////////////////////////////
	float& Tensor::operator()(int w, int h, int d, int n)
	{
		return Get(w, h, d, n);
	}

    //////////////////////////////////////////////////////////////////////////
    float Tensor::operator()(int w, int h, int d, int n) const
    {
        return Get(w, h, d, n);
    }

	//////////////////////////////////////////////////////////////////////////
	float Tensor::TryGet(float def, int w, int h, int d, int n) const
	{
		if (h < 0 || h >= Height() || w < 0 || w >= Width() || d < 0 || d >= Depth())
			return def;

		return Get(w, h, d, n);
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::SetFlat(float value, int i)
	{
		CopyToHost();
		m_Values[i] = value;
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::Set(float value, int w, int h, int d, int n)
	{
		CopyToHost();
		m_Values[m_Shape.GetIndex(w, h, d, n)] = value;
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::TrySet(float value, int w, int h, int d, int n)
	{
		if (h < 0 || h >= Height() || w < 0 || w >= Width() || d < 0 || d >= Depth() || n < 0 || n > Batch())
			return;

		Set(value, w, h, d, n);
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::CopyTo(Tensor& result, float tau) const
	{
		CopyToHost();
		//if (m_Shape.Length != result.m_Shape.Length) throw new Exception("Incompatible tensors.");

		if (tau <= 0)
			copy(m_Values.begin(), m_Values.end(), result.m_Values.begin());
		else
			Map([&](float v1, float v2) { return v1 * tau + v2 * (1 - tau); }, result, result);
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::CopyBatchTo(int batchId, int targetBatchId, Tensor& result) const
	{
		CopyToHost();
		result.OverrideHost();
		//if (m_Shape.Width != result.m_Shape.Width || m_Shape.Height != result.m_Shape.Height || m_Shape.Depth != result.m_Shape.Depth) throw new Exception("Incompatible tensors.");

		copy(m_Values.begin() + batchId * m_Shape.Dim0Dim1Dim2, 
			 m_Values.begin() + batchId * m_Shape.Dim0Dim1Dim2 + m_Shape.Dim0Dim1Dim2,
			 result.m_Values.begin() + targetBatchId * m_Shape.Dim0Dim1Dim2);
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::CopyDepthTo(int depthId, int batchId, int targetDepthId, int targetBatchId, Tensor& result) const
	{
		CopyToHost();
		result.OverrideHost();
		//if (m_Shape.Width != result.m_Shape.Width || m_Shape.Height != result.m_Shape.Height) throw new Exception("Incompatible tensors.");

		copy(m_Values.begin() + batchId * m_Shape.Dim0Dim1Dim2 + depthId * m_Shape.Dim0Dim1, 
			 m_Values.begin() + batchId * m_Shape.Dim0Dim1Dim2 + depthId * m_Shape.Dim0Dim1 + m_Shape.Dim0Dim1,
		     result.m_Values.begin() + targetBatchId * m_Shape.Dim0Dim1Dim2 + targetDepthId * m_Shape.Dim0Dim1);
	}

	//////////////////////////////////////////////////////////////////////////
	Tensor Tensor::GetBatch(int batchId) const
	{
		Tensor result(Shape(Width(), Height(), Depth()));
		CopyBatchTo(batchId, 0, result);
		return result;
	}

	//////////////////////////////////////////////////////////////////////////
	Tensor Tensor::GetDepth(int depthId, int batchId) const
	{
		Tensor result(Shape(Width(), Height()));
		CopyDepthTo(depthId, batchId, 0, 0, result);
		return result;
	}

	//////////////////////////////////////////////////////////////////////////
	bool Tensor::Equals(const Tensor& other, float epsilon) const
	{
		CopyToHost();
		other.CopyToHost();

		//assert(Values.size() == other.Values.size(), "Comparing tensors with different number of elements!");
		if (m_Values.size() != other.m_Values.size())
			return false;

		if (epsilon == 0)
			return m_Values == other.m_Values;

		for (int i = 0; i < m_Values.size(); ++i)
			if (abs(m_Values[i] - other.m_Values[i]) > epsilon)
				return false;

		return true;
	}

	//////////////////////////////////////////////////////////////////////////
	float Tensor::GetMaxData(int batch, int& maxIndex) const
	{
		CopyToHost();
		maxIndex = -1;
		float maxValue = numeric_limits<float>().min();

		if (batch < 0)
		{
			for (int i = 0; i < m_Values.size(); ++i)
				if (m_Values[i] > maxValue)
				{
					maxValue = m_Values[i];
					maxIndex = i;
				}
		}
		else
		{
			int batchLen = BatchLength();

			for (int i = 0, idx = batch * batchLen; i < batchLen; ++i, ++idx)
				if (m_Values[idx] > maxValue)
				{
					maxValue = m_Values[idx];
					maxIndex = i;
				}
		}

		return maxValue;
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::Elu(float alpha, Tensor& result) const
	{
		m_Op->Elu(*this, alpha, result);
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::EluGradient(const Tensor& output, const Tensor& outputGradient, float alpha, Tensor& result) const
	{
		m_Op->EluGradient(output, outputGradient, alpha, result);
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::Softmax(Tensor& result) const
	{
		m_Op->Softmax(*this, result);
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::SoftmaxGradient(const Tensor& output, const Tensor& outputGradient, Tensor& result) const
	{
		m_Op->SoftmaxGradient(output, outputGradient, result);
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::CopyToDevice() const
	{
		if (m_CurrentLocation == ELocation::Device)
			return;
        
		GetDeviceVar().CopyToDevice(m_Values);
		m_CurrentLocation = ELocation::Device;
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::CopyToHost() const
	{
		if (m_CurrentLocation == ELocation::Host)
			return;

		GetDeviceVar().CopyToHost(m_Values);
		m_CurrentLocation = ELocation::Host;
	}

    //////////////////////////////////////////////////////////////////////////
    void Tensor::OverrideHost() const
    {
        m_CurrentLocation = ELocation::Host;
    }

    //////////////////////////////////////////////////////////////////////////
    const Neuro::CudaDeviceVariable<float>& Tensor::GetDeviceVar() const
    {
        if (!m_GpuData.m_DeviceVar)
            m_GpuData.m_DeviceVar = new CudaDeviceVariable<float>(m_Values.size());
        return *m_GpuData.m_DeviceVar;
    }

    //////////////////////////////////////////////////////////////////////////
    const float* Tensor::GetDevicePtr() const
    {
        return static_cast<const float*>(GetDeviceVar().GetDevicePtr());
    }

    //////////////////////////////////////////////////////////////////////////
    float* Tensor::GetDevicePtr()
    {
        return static_cast<float*>(GetDeviceVar().GetDevicePtr());
    }

    //////////////////////////////////////////////////////////////////////////
	Neuro::TensorOpCpu* Tensor::GetOpMode(EOpMode mode)
	{
		switch (mode)
		{
		case EOpMode::CPU:
			return g_OpCpu;
        case EOpMode::MultiCPU:
			return g_OpMultiCpu = (g_OpMultiCpu ? g_OpMultiCpu : new TensorOpMultiCpu());
        case EOpMode::GPU:
			return g_OpGpu = (g_OpGpu ? g_OpGpu : new TensorOpGpu());
		}

		return nullptr;
	}

    //////////////////////////////////////////////////////////////////////////
    Tensor::GPUData::~GPUData()
    {
        delete(m_DeviceVar);
        delete(m_ConvWorkspace);
        delete(m_ConvBackWorkspace);
        delete(m_ConvBackKernelWorkspace);
    }

    //////////////////////////////////////////////////////////////////////////
    void Tensor::GPUData::UpdateWorkspace(CudaDeviceVariable<char>*& workspace, size_t size)
    {
        if (workspace && workspace->GetSizeInBytes() != size)
        {
            delete workspace;
            workspace = nullptr;
        }

        if (!workspace)
            workspace = new CudaDeviceVariable<char>(size);
    }
}

