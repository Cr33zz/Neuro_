#include "Tensors/Tensor.h"
#include "Tensors/TensorOpCpu.h"
#include "Random.h"
#include "Tools.h"

namespace Neuro
{
	//////////////////////////////////////////////////////////////////////////
	Tensor::Tensor()
	{
		CurrentLocation = ELocation::Host;
		SetOpMode(EOpMode::CPU);
	}

	//////////////////////////////////////////////////////////////////////////
	Tensor::Tensor(const Shape& shape)
		: Tensor()
	{
		m_Shape = shape;
		Values.resize(shape.Length);
	}

	//////////////////////////////////////////////////////////////////////////
	Tensor::Tensor(const Tensor& t)
		: Tensor()
	{
		t.CopyToHost();
		m_Shape = t.GetShape();
		Values = t.Values;
	}

	//////////////////////////////////////////////////////////////////////////
	Tensor::Tensor(const vector<float>& values)
		: Tensor()
	{
		m_Shape = Shape((int)values.size());
		Values = values;
	}

	//////////////////////////////////////////////////////////////////////////
	Tensor::Tensor(const vector<float>& values, const Shape& shape)
		: Tensor()
	{
		//assert(Values.size() == shape.Length && "Invalid array size {Values.size()}. Expected {shape.Length}.");
		m_Shape = shape;
		Values = values;
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
	void Tensor::SetOpMode(EOpMode mode)
	{
		switch (mode)
		{
		case EOpMode::CPU:
			Op = (g_OpCpu ? g_OpCpu : g_OpCpu = new TensorOpCpu());
			return;
		/*case EOpMode.MultiCPU:
			Op = new TensorOpMultiCpu();
			return;
		case EOpMode.GPU:
			Op = new TensorOpGpu();
			return;*/
		}
	}

	//////////////////////////////////////////////////////////////////////////
	std::vector<float>& Tensor::GetValues()
	{
		CopyToHost();
		return Values;
	}

	//////////////////////////////////////////////////////////////////////////
	Tensor& Tensor::FillWithRand(int seed, float min, float max)
	{
		CurrentLocation = ELocation::Host;

		auto fillUp = [&](Random& rng)
		{
			for (int i = 0; i < Values.size(); ++i)
				Values[i] = min + (max - min) * rng.NextFloat();
		};

		if (seed > 0)
		{
			Random tempRng(seed);
			fillUp(tempRng);
		}
		else
		{
			fillUp(g_Rng);
		}

		return *this;
	}

	//////////////////////////////////////////////////////////////////////////
	Tensor& Tensor::FillWithRange(float start, float increment)
	{
		CurrentLocation = ELocation::Host;
		for (int i = 0; i < Values.size(); ++i)
			Values[i] = start + i * increment;
		return *this;
	}

	//////////////////////////////////////////////////////////////////////////
	Tensor& Tensor::FillWithValue(float value)
	{
		CurrentLocation = ELocation::Host;
		for (int i = 0; i < Values.size(); ++i)
			Values[i] = value;
		return *this;
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::Zero()
	{
		CurrentLocation = ELocation::Host;
		fill(Values.begin(), Values.end(), 0);
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::Mul(bool transposeT, const Tensor& t, Tensor& result)
	{
		assert((!transposeT && Width() == t.Height()) || (transposeT && Width() == t.Width()));
		assert(t.Depth() == Depth());

		Op->Mul(false, transposeT, *this, t, result);
	}

	//////////////////////////////////////////////////////////////////////////
	Tensor Tensor::Mul(bool transposeT, const Tensor& t)
	{
		Tensor result(Shape(transposeT ? t.m_Shape.Height() : t.m_Shape.Width(), Height(), Depth(), BatchSize()));
		Mul(transposeT, t, result);
		return result;
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::Mul(const Tensor& t, Tensor& result)
	{
		Mul(false, t, result);
	}

	//////////////////////////////////////////////////////////////////////////
	Tensor Tensor::Mul(const Tensor& t)
	{
		return Mul(false, t);
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::MulElem(const Tensor& t, Tensor& result)
	{
		assert(SameDimensionsExceptBatches(t));
		assert(t.BatchSize() == result.BatchSize());

		Op->MulElem(*this, t, result);
	}


	//////////////////////////////////////////////////////////////////////////
	Tensor Tensor::MulElem(const Tensor& t)
	{
		Tensor result(m_Shape);
		MulElem(t, result);
		return result;
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::Mul(float v, Tensor& result)
	{
		CopyToHost();
		result.CurrentLocation = ELocation::Host;

		for (int i = 0; i < Values.size(); ++i)
			result.Values[i] = Values[i] * v;
	}

	//////////////////////////////////////////////////////////////////////////
	Tensor Tensor::Mul(float v)
	{
		Tensor result(m_Shape);
		Mul(v, result);
		return result;
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::Div(const Tensor& t, Tensor& result)
	{
		CopyToHost();
		result.CurrentLocation = ELocation::Host;

		assert(SameDimensionsExceptBatches(t));
		assert(t.BatchSize() == result.BatchSize());

		for (int i = 0; i < Values.size(); ++i)
			result.Values[i] = Values[i] / t.Values[i];
	}


	//////////////////////////////////////////////////////////////////////////
	Tensor Tensor::Div(const Tensor& t)
	{
		Tensor result(m_Shape);
		Div(t, result);
		return result;
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::Div(float v, Tensor& result)
	{
		CopyToHost();
		result.CurrentLocation = ELocation::Host;

		for (int i = 0; i < Values.size(); ++i)
			result.Values[i] = Values[i] / v;
	}

	//////////////////////////////////////////////////////////////////////////
	Tensor Tensor::Div(float v)
	{
		Tensor result(m_Shape);
		Div(v, result);
		return result;
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::Add(float alpha, float beta, const Tensor& t, Tensor& result)
	{
		assert(SameDimensionsExceptBatches(t));
		assert(t.BatchSize() == result.BatchSize() || t.BatchSize() == 1);

		Op->Add(alpha, *this, beta, t, result);
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::Add(const Tensor& t, Tensor& result)
	{
		Add(1, 1, t, result);
	}

	//////////////////////////////////////////////////////////////////////////
	Tensor Tensor::Add(const Tensor& t)
	{
		Tensor result(m_Shape);
		Add(t, result);
		return result;
	}

	//////////////////////////////////////////////////////////////////////////
	Tensor Tensor::Add(float alpha, float beta, const Tensor& t)
	{
		Tensor result(m_Shape);
		Add(alpha, beta, t, result);
		return result;
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::Add(float v, Tensor& result)
	{
		CopyToHost();
		for (int i = 0; i < Values.size(); ++i)
			result.Values[i] = Values[i] + v;
	}

	//////////////////////////////////////////////////////////////////////////
	Tensor Tensor::Add(float v)
	{
		Tensor result(m_Shape);
		Add(v, result);
		return result;
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::Sub(const Tensor& t, Tensor& result)
	{
		assert(SameDimensionsExceptBatches(t));
		assert(t.BatchSize() == result.BatchSize() || t.BatchSize() == 1);

		Op->Sub(*this, t, result);
	}

	//////////////////////////////////////////////////////////////////////////
	Tensor Tensor::Sub(const Tensor& t)
	{
		Tensor result(m_Shape);
		Sub(t, result);
		return result;
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::Sub(float v, Tensor& result)
	{
		CopyToHost();
		for (int i = 0; i < Values.size(); ++i)
			result.Values[i] = Values[i] - v;
	}

	//////////////////////////////////////////////////////////////////////////
	Tensor Tensor::Sub(float v)
	{
		Tensor result(m_Shape);
		Sub(v, result);
		return result;
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::Negated(Tensor& result)
	{
		for (int i = 0; i < Values.size(); ++i)
			result.Values[i] = -Values[i];
	}

	//////////////////////////////////////////////////////////////////////////
	Tensor Tensor::Negated()
	{
		Tensor result(m_Shape);
		Negated(result);
		return result;
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::Clipped(float min, float max, Tensor& result)
	{
		Map([&](float x) { return Tools::Clip(x, min, max); }, result);
	}

	//////////////////////////////////////////////////////////////////////////
	Tensor Tensor::Clipped(float min, float max)
	{
		Tensor result(m_Shape);
		Clipped(min, max, result);
		return result;
	}

	//////////////////////////////////////////////////////////////////////////
	Tensor Tensor::DiagFlat()
	{
		Tensor result(Shape(BatchLength(), BatchLength(), 1, BatchSize()));

		int batchLen = BatchLength();

		for (int b = 0; b < BatchSize(); ++b)
			for (int i = 0; i < batchLen; ++i)
				result(i, i, 0, b) = Values[b * batchLen + i];

		return result;
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::Map(const function<float(float)>& func, Tensor& result)
	{
		Op->Map(func, *this, result);
	}

	//////////////////////////////////////////////////////////////////////////
	Tensor Tensor::Map(const function<float(float)>& func)
	{
		Tensor result(m_Shape);
		Map(func, result);
		return result;
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::Map(const function<float(float, float)>& func, const Tensor& other, Tensor& result)
	{
		Op->Map(func, *this, other, result);
	}

	//////////////////////////////////////////////////////////////////////////
	Tensor Tensor::Map(const function<float(float, float)>& func, const Tensor& other)
	{
		Tensor result(m_Shape);
		Map(func, other, result);
		return result;
	}

	//////////////////////////////////////////////////////////////////////////
	Tensor Tensor::SumBatches()
	{
		Tensor result(Shape(m_Shape.Width(), m_Shape.Height(), m_Shape.Depth(), 1));
		Op->SumBatches(*this, result);
		return result;
	}

	//////////////////////////////////////////////////////////////////////////
	float Tensor::Sum(int batch)
	{
		CopyToHost();
		int batchLen = batch < 0 ? Length() : BatchLength();
		float sum = 0;

		for (int i = 0, idx = batch * batchLen; i < batchLen; ++i, ++idx)
			sum += Values[idx];

		return sum;
	}

	//////////////////////////////////////////////////////////////////////////
	Tensor Tensor::SumPerBatch()
	{
		CopyToHost();
		Tensor result(Shape(1, 1, 1, m_Shape.BatchSize()));

		for (int n = 0; n < BatchSize(); ++n)
			result.Values[n] = Sum(n);

		return result;
	}

	//////////////////////////////////////////////////////////////////////////
	Tensor Tensor::AvgBatches()
	{
		CopyToHost();
		Tensor result = SumBatches();

		int batchLen = BatchLength();

		for (int n = 0; n < batchLen; ++n)
			result.Values[n] /= BatchSize();

		return result;
	}

	//////////////////////////////////////////////////////////////////////////
	float Tensor::Avg(int batch)
	{
		return Sum(batch) / (batch < 0 ? Length() : BatchLength());
	}

	//////////////////////////////////////////////////////////////////////////
	Tensor Tensor::AvgPerBatch()
	{
		Tensor result = SumPerBatch();

		int batchLen = BatchLength();

		for (int n = 0; n < BatchSize(); ++n)
			result.Values[n] /= batchLen;

		return result;
	}

	//////////////////////////////////////////////////////////////////////////
	float Tensor::Max(int batch)
	{
		int maxIndex;
		return GetMaxData(batch, maxIndex);
	}

	//////////////////////////////////////////////////////////////////////////
	Tensor Tensor::MaxPerBatch()
	{
		Tensor result(Shape(1, 1, 1, m_Shape.BatchSize()));
		for (int n = 0; n < BatchSize(); ++n)
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
			copy(t.Values.begin(), t.Values.end(), output.Values.begin() + t.Length() * n);
		}

		return output;
	}

	//////////////////////////////////////////////////////////////////////////
	Tensor Tensor::MergeIntoDepth(const vector<Tensor>& tensors, int forcedDepth /*= 0*/)
	{
		/*if (tensors.Count == 0)
			throw new Exception("List cannot be empty.");*/

		Tensor output(Shape(tensors[0].Width, tensors[0].Height, max(tensors.size(), forcedDepth)));

		const Tensor& t = tensors[0];
		t.CopyToHost();

		int t0_copies = forcedDepth > 0 ? forcedDepth - tensors.size() : 0;

		for (int n = 0; n < t0_copies; ++n)
		{
			copy(t.Values.begin(), t.Values.end(), output.Values.begin() + t.Length * n);
		}

		for (int n = t0_copies; n < output.Depth; ++n)
		{
			t = tensors[n - t0_copies];
			t.CopyToHost();
			copy(t.Values.begin(), t.Values.end(), output.Values.begin() + t.Length * n);
		}

		return output;
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::Concat(const vector<Tensor>& inputs, Tensor& result)
	{
		for (int b = 0; b < result.BatchSize(); ++b)
		{
			int elementsCopied = 0;
			for (int i = 0; i < inputs.size(); ++i)
			{
				inputs[i].CopyToHost();
				copy(inputs[i].Values.begin() + b * inputs[i].BatchLength(), inputs[i].Values.begin() + inputs[i].BatchLength(), result.Values.begin() + b * result.BatchLength() + elementsCopied);
				elementsCopied += inputs[i].BatchLength();
			}
		}
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::Split(vector<Tensor>& outputs)
	{
		CopyToHost();
		for (int b = 0; b < BatchSize(); ++b)
		{
			int elementsCopied = 0;
			for (int i = 0; i < outputs.size(); ++i)
			{
				outputs[i].CopyToHost();
				copy(Values.begin() + b * BatchLength() + elementsCopied, Values.begin() + outputs[i].BatchLength(), outputs[i].Values.begin() + b * outputs[i].BatchLength());
				elementsCopied += outputs[i].BatchLength();
			}
		}
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::MergeMin(const vector<Tensor>& inputs, Tensor& result)
	{
		inputs[0].CopyTo(result);
		for (int i = 1; i < inputs.size(); ++i)
			for (int j = 0; j < result.Length; ++j)
				result.Values[j] = result.Values[j] > inputs[i].Values[j] ? inputs[i].Values[j] : result.Values[j];
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::MergeMax(const vector<Tensor>& inputs, Tensor& result)
	{
		inputs[0].CopyTo(result);
		for (int i = 1; i < inputs.size(); ++i)
			for (int j = 0; j < result.Length; ++j)
				result.Values[j] = result.Values[j] < inputs[i].Values[j] ? inputs[i].Values[j] : result.Values[j];
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::MergeSum(const vector<Tensor>& inputs, Tensor& result)
	{
		result.Zero();
		for (int i = 0; i < inputs.size(); ++i)
			for (int j = 0; j < result.Length; ++j)
				result.Values[j] += inputs[i].Values[j];
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::MergeAvg(const vector<Tensor>& inputs, Tensor& result)
	{
		MergeSum(inputs, result);
		result.Div(inputs.size(), result);
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::MergeMinMaxGradient(Tensor output, const vector<Tensor>& inputs, Tensor outputGradient, vector<Tensor>& results)
	{
		for (int i = 0; i < inputs.size(); ++i)
		{
			results[i].Zero();
			for (int j = 0; j < output.Length; ++j)
				results[i].Values[j] = inputs[i].Values[j] == output.Values[j] ? outputGradient.Values[j] : 0;
		}
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::MergeSumGradient(Tensor output, const vector<Tensor>& inputs, Tensor outputGradient, vector<Tensor>& results)
	{
		for (int i = 0; i < inputs.size(); ++i)
			outputGradient.CopyTo(results[i]);
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::MergeAvgGradient(Tensor output, const vector<Tensor>& inputs, Tensor outputGradient, vector<Tensor>& results)
	{
		MergeSumGradient(output, inputs, outputGradient, results);
		for (int i = 0; i < results.size(); ++i)
			results[i].Div(results.size(), results[i]);
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::Normalized(Tensor& result)
	{
		float sum = Sum();
		Map(x = > x / sum, result);
	}

	//////////////////////////////////////////////////////////////////////////
	int Tensor::ArgMax(int batch /*= -1*/)
	{
		int maxIndex;
		GetMaxData(batch, out maxIndex);
		return maxIndex;
	}

	//////////////////////////////////////////////////////////////////////////
	Tensor Tensor::ArgMaxPerBatch()
	{
		var result(Shape(1, 1, 1, m_Shape.BatchSize()));
		for (int n = 0; n < BatchSize(); ++n)
			result[0, 0, 0, n] = ArgMax(n);
		return result;
	}

	//////////////////////////////////////////////////////////////////////////
	const Tensor& Tensor::transposed()
	{
		Tensor result(Shape(Height, Width, Depth, BatchSize()));
		Transpose(result);

		return result;
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::Transpose(Tensor& result)
	{
		Op->Transpose(this, result);
	}

	//////////////////////////////////////////////////////////////////////////
	Tensor Tensor::Reshaped(Shape shape)
	{
		return Tensor(Values, m_Shape.Reshaped({ shape.Width, shape.Height, shape.Depth, shape.BatchSize() }));
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::Reshape(Shape shape)
	{
		Shape = m_Shape.Reshaped(new[] { shape.Width, shape.Height, shape.Depth, shape.BatchSize() });
	}

	//////////////////////////////////////////////////////////////////////////
	Tensor Tensor::Resized(int width, int height /*= 1*/, int depth /*= 1*/)
	{
		int newBatchLength = width * height * depth;
		Tensor result(Shape(width, height, depth, m_Shape.BatchSize()));
		for (int n = 0; n < BatchSize(); ++n)
			for (int i = 0, idx = n * newBatchLength; i < newBatchLength; ++i, ++idx)
				result.Values[idx] = Values[n * BatchLength() + i % BatchLength()];
		return result;
	}

	//////////////////////////////////////////////////////////////////////////
	Tensor Tensor::FlattenHoriz()
	{
		return Reshaped(m_Shape.Reshaped(new[] { Shape.Auto, 1, 1, m_Shape.BatchSize() }));
	}

	//////////////////////////////////////////////////////////////////////////
	Tensor Tensor::FlattenVert()
	{
		return Reshaped(m_Shape.Reshaped(new[] { 1, Shape.Auto, 1, m_Shape.BatchSize() }));
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::Rotated180(Tensor& result)
	{
		assert(SameDimensionsExceptBatches(result));

		for (int n = 0; n < BatchSize(); ++n)
			for (int d = 0; d < Depth(); ++d)
				for (int h = Height() - 1; h >= 0; --h)
					for (int w = Width() - 1; w >= 0; --w)
						result.Set(Get(Width() - w - 1, Height() - h - 1, d, n), w, h, d, n);
	}

	//////////////////////////////////////////////////////////////////////////
	Tensor Tensor::Rotated180()
	{
		Tensor result(m_Shape);
		Rotated180(result);
		return result;
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::Conv2D(Tensor kernels, int stride, EPaddingType padding, Tensor& result)
	{
		assert(Depth == kernels.Depth);

		Op->Conv2D(this, kernels, stride, padding, result);
	}

	//////////////////////////////////////////////////////////////////////////
	Tensor Tensor::Conv2D(Tensor kernels, int stride, EPaddingType padding)
	{
		int outputWidth = 0, outputHeight = 0, paddingX = 0, paddingY = 0;
		GetPaddingParams(padding, Width(), Height(), kernels.Width(), kernels.Height(), stride, outputHeight, outputWidth, paddingX, paddingY);

		Tensor result(Shape(outputWidth, outputHeight, kernels.BatchSize(), BatchSize()));
		Conv2D(kernels, stride, padding, result);
		return result;
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::Conv2DInputsGradient(Tensor gradient, Tensor kernels, int stride, EPaddingType padding, Tensor inputsGradient)
	{
		inputsGradient.Zero();
		Op->Conv2DInputGradient(gradient, kernels, stride, padding, inputsGradient);
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::Conv2DKernelsGradient(Tensor input, Tensor gradient, int stride, EPaddingType padding, Tensor kernelsGradient)
	{
		kernelsGradient.Zero();
		Op->Conv2DKernelsGradient(input, gradient, stride, padding, kernelsGradient);
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::Conv2DGradient_old(Tensor input, Tensor kernels, Tensor gradient, int stride, EPaddingType padding, Tensor inputGradient, Tensor kernelsGradient)
	{
		inputGradient.Zero();
		kernelsGradient.Zero();
		int outputWidth = 0, outputHeight = 0, paddingX = 0, paddingY = 0;
		GetPaddingParams(padding, input.Width(), input.Height(), kernels.Width(), kernels.Height(), stride, outputHeight, outputWidth, paddingX, paddingY);
		Op->Conv2DGradient_old(input, kernels, gradient, stride, paddingX, paddingY, inputGradient, kernelsGradient);
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::Pool(int filterSize, int stride, EPoolType type, EPaddingType padding, Tensor& result)
	{
		int outWidth = 0, outHeight = 0, paddingX = 0, paddingY = 0;
		GetPaddingParams(padding, Width(), Height(), filterSize, filterSize, stride, outHeight, outWidth, paddingX, paddingY);

		assert(result.Width == outWidth);
		assert(result.Height == outHeight);
		assert(result.BatchSize() == BatchSize());

		Op->Pool(this, filterSize, stride, type, paddingX, paddingY, result);
	}

	//////////////////////////////////////////////////////////////////////////
	Tensor Tensor::Pool(int filterSize, int stride, EPoolType type, EPaddingType padding)
	{
		int outWidth = 0, outHeight = 0, paddingX = 0, paddingY = 0;
		GetPaddingParams(padding, Width(), Height(), filterSize, filterSize, stride, outHeight, outWidth, paddingX, paddingY);

		Tensor result(Shape(outWidth, outHeight, Depth(), BatchSize()));
		Pool(filterSize, stride, type, padding, result);

		return result;
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::PoolGradient(Tensor output, Tensor input, Tensor outputGradient, int filterSize, int stride, EPoolType type, EPaddingType padding, Tensor& result)
	{
		assert(output.SameDimensionsExceptBatches(outputGradient));

		int outWidth = 0, outHeight = 0, paddingX = 0, paddingY = 0;
		GetPaddingParams(padding, result.Width, result.Height, filterSize, filterSize, stride, out outHeight, out outWidth, out paddingX, out paddingY);

		Op->PoolGradient(output, input, outputGradient, filterSize, stride, type, paddingX, paddingY, result);
	}

	//////////////////////////////////////////////////////////////////////////
	std::string Tensor::ToString()
	{
		string s = "";
		for (int n = 0; n < BatchSize(); ++n)
		{
			if (BatchSize() > 1)
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

			if (BatchSize() > 1)
				s += "}" + (n < BatchSize() - 1 ? "\n" : "");
		}

		return s;
	}

	//////////////////////////////////////////////////////////////////////////
	bool Tensor::SameDimensionsExceptBatches(const Tensor& t)
	{
		return Width == t.Width && Height == t.Height && Depth == t.Depth;
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
		return Values[i];
	}

	//////////////////////////////////////////////////////////////////////////
	float& Tensor::Get(int w, int h /*= 0*/, int d /*= 0*/, int n /*= 0*/)
	{
		CopyToHost();
		return Values[m_Shape.GetIndex(w, h, d, n)];
	}

	//////////////////////////////////////////////////////////////////////////
	float& Tensor::operator()(int w, int h /*= 0*/, int d /*= 0*/, int n /*= 0*/)
	{
		return Get(w, h, d, n);
	}

	//////////////////////////////////////////////////////////////////////////
	float Tensor::TryGet(float def, int w, int h /*= 0*/, int d /*= 0*/, int n /*= 0*/)
	{
		if (h < 0 || h >= Height || w < 0 || w >= Width || d < 0 || d >= Depth)
			return def;

		return Get(w, h, d, n);
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::SetFlat(float value, int i)
	{
		CopyToHost();
		Values[i] = value;
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::Set(float value, int w, int h /*= 0*/, int d /*= 0*/, int n /*= 0*/)
	{
		CopyToHost();
		Values[m_Shape.GetIndex(w, h, d, n)] = value;
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::TrySet(float value, int w, int h /*= 0*/, int d /*= 0*/, int n /*= 0*/)
	{
		if (h < 0 || h >= Height() || w < 0 || w >= Width() || d < 0 || d >= Depth() || n < 0 || n > BatchSize())
			return;

		Set(value, w, h, d, n);
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::CopyTo(Tensor& result, float tau) const
	{
		CopyToHost();
		//if (m_Shape.Length != result.m_Shape.Length) throw new Exception("Incompatible tensors.");

		if (tau > 0)
			copy(Values.begin(), Values.end(), result.Values.begin());
		else
			Map([&](float v1, float v2) { return v1 * tau + v2 * (1 - tau); }, result, result);
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::CopyBatchTo(int batchId, int targetBatchId, Tensor& result)
	{
		CopyToHost();
		result.CurrentLocation = ELocation::Host;
		//if (m_Shape.Width != result.m_Shape.Width || m_Shape.Height != result.m_Shape.Height || m_Shape.Depth != result.m_Shape.Depth) throw new Exception("Incompatible tensors.");

		copy(Values.begin() + batchId * m_Shape.Dim0Dim1Dim2, 
			 Values.begin() + batchId * m_Shape.Dim0Dim1Dim2 + m_Shape.Dim0Dim1Dim2,
			 result.Values.begin() + targetBatchId * m_Shape.Dim0Dim1Dim2);
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::CopyDepthTo(int depthId, int batchId, int targetDepthId, int targetBatchId, Tensor& result)
	{
		CopyToHost();
		result.CurrentLocation = ELocation::Host;
		//if (m_Shape.Width != result.m_Shape.Width || m_Shape.Height != result.m_Shape.Height) throw new Exception("Incompatible tensors.");

		copy(Values.begin() + batchId * m_Shape.Dim0Dim1Dim2 + depthId * m_Shape.Dim0Dim1, 
			 Values.begin() + batchId * m_Shape.Dim0Dim1Dim2 + depthId * m_Shape.Dim0Dim1 + m_Shape.Dim0Dim1,
		     result.Values.begin() + targetBatchId * m_Shape.Dim0Dim1Dim2 + targetDepthId * m_Shape.Dim0Dim1);
	}

	//////////////////////////////////////////////////////////////////////////
	Tensor Tensor::GetBatch(int batchId)
	{
		Tensor result(Shape(Width(), Height(), Depth()));
		CopyBatchTo(batchId, 0, result);
		return result;
	}

	//////////////////////////////////////////////////////////////////////////
	Tensor Tensor::GetDepth(int depthId, int batchId)
	{
		Tensor result(Shape(Width(), Height()));
		CopyDepthTo(depthId, batchId, 0, 0, result);
		return result;
	}

	//////////////////////////////////////////////////////////////////////////
	bool Tensor::Equals(const Tensor& other, float epsilon)
	{
		CopyToHost();
		other.CopyToHost();

		//assert(Values.size() == other.Values.size(), "Comparing tensors with different number of elements!");
		if (Values.size() != other.Values.size())
			return false;

		if (epsilon == 0)
			return Values == other.Values;

		for (int i = 0; i < Values.size(); ++i)
			if (abs(Values[i] - other.Values[i]) > epsilon)
				return false;

		return true;
	}

	//////////////////////////////////////////////////////////////////////////
	float Tensor::GetMaxData(int batch, int& maxIndex)
	{
		CopyToHost();
		maxIndex = -1;
		float maxValue = numeric_limits<float>().min();

		if (batch < 0)
		{
			for (int i = 0; i < Values.size(); ++i)
				if (Values[i] > maxValue)
				{
					maxValue = Values[i];
					maxIndex = i;
				}
		}
		else
		{
			int batchLen = BatchLength();

			for (int i = 0, idx = batch * batchLen; i < batchLen; ++i, ++idx)
				if (Values[idx] > maxValue)
				{
					maxValue = Values[idx];
					maxIndex = i;
				}
		}

		return maxValue;
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::Elu(float alpha, Tensor& result)
	{
		Op->Elu(this, alpha, result);
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::EluGradient(Tensor output, Tensor outputGradient, float alpha, Tensor& result)
	{
		Op->EluGradient(output, outputGradient, alpha, result);
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::Softmax(Tensor& result)
	{
		Op->Softmax(this, result);
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::SoftmaxGradient(Tensor output, Tensor outputGradient, Tensor& result)
	{
		Op->SoftmaxGradient(output, outputGradient, result);
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::CopyToDevice() const
	{
		if (CurrentLocation == ELocation::Device)
			return;

		//GpuData.DeviceVar = GpuData.DeviceVar ?? new CudaDeviceVariable<float>(m_Shape.Length);
		//GpuData.DeviceVar.CopyToDevice(Values);
		CurrentLocation = ELocation::Device;
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::CopyToHost() const
	{
		if (CurrentLocation == ELocation::Host)
			return;

		//GpuData.DeviceVar.CopyToHost(Values);
		CurrentLocation = ELocation::Host;
	}
}

