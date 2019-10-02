#include <algorithm>
#include <fstream>
#include <numeric>
#include <FreeImage.h>

#include "Tensors/Tensor.h"
#include "Tensors/Cuda/CudaDeviceVariable.h"
#include "Tensors/TensorOpCpu.h"
#include "Tensors/TensorOpMultiCpu.h"
#include "Tensors/TensorOpGpu.h"
#include "Tensors/TensorFormatter.h"
#include "Random.h"
#include "Tools.h"

namespace Neuro
{
    using namespace std;

	TensorOpCpu* Tensor::g_OpCpu = new TensorOpCpu();
    TensorOpCpu* Tensor::g_OpMultiCpu = nullptr;
    TensorOpCpu* Tensor::g_OpGpu = nullptr;

    TensorOpCpu* Tensor::g_DefaultOp = nullptr;
    TensorOpCpu* Tensor::g_ForcedOp = nullptr;

    FREE_IMAGE_FORMAT GetFormat(const string& fileName)
    {
        auto fif = FreeImage_GetFileType(fileName.c_str());
        
        if (fif == FIF_UNKNOWN)
            fif = FreeImage_GetFIFFromFilename(fileName.c_str());

        return fif;
    }

    //////////////////////////////////////////////////////////////////////////
    Tensor::Tensor(const Shape& shape, const string& name)
        : Tensor(name)
    {
        m_Shape = shape;
        m_Values.resize(shape.Length);
    }

    //////////////////////////////////////////////////////////////////////////
    Tensor::Tensor(istream& stream)
        : Tensor("")
    {
        LoadBin(stream);
    }

    //////////////////////////////////////////////////////////////////////////
    Tensor::Tensor(const string& name)
        : m_Name(name), m_Shape(0)
	{
		OverrideHost();
        m_Values.resize(m_Shape.Length);
		m_Op = DefaultOp();
	}

	//////////////////////////////////////////////////////////////////////////
	Tensor::Tensor(const Tensor& t)
	{
        *this = t;
	}

    //////////////////////////////////////////////////////////////////////////
    Tensor& Tensor::operator=(const Tensor& t)
    {
        if (this != &t)
        {
            t.CopyToHost();
            m_GpuData.Release();
            OverrideHost();
            m_Values = t.m_Values;
            m_Name = t.m_Name;
            m_Shape = t.m_Shape;
            m_Op = t.m_Op;
        }
        return *this;
    }

	//////////////////////////////////////////////////////////////////////////
	Tensor::Tensor(const vector<float>& values, const string& name)
		: Tensor(name)
	{
		m_Shape = Shape((int)values.size());
		m_Values = values;
	}

	//////////////////////////////////////////////////////////////////////////
	Tensor::Tensor(const vector<float>& values, const Shape& shape, const string& name)
		: Tensor(name)
	{
		assert(values.size() == shape.Length);// && string("Invalid array size ") + to_string(values.size()) + ". Expected " + to_string(shape.Length) + ".");
		m_Shape = shape;
		m_Values = values;
	}

	//////////////////////////////////////////////////////////////////////////
	Tensor::Tensor(const string& imageFile, bool normalize, bool grayScale, const string& name)
        : Tensor(name)
	{
        ImageLibInit();

        auto format = GetFormat(imageFile);
        assert(format != FIF_UNKNOWN);

        FIBITMAP* image = FreeImage_Load(format, imageFile.c_str());

        assert(image);

        const uint32_t WIDTH = FreeImage_GetWidth(image);
        const uint32_t HEIGHT = FreeImage_GetHeight(image);

        m_Shape = Shape(WIDTH, HEIGHT, grayScale ? 1 : 3);
        m_Values.resize(m_Shape.Length);

        RGBQUAD color;

        for (uint32_t h = 0; h < HEIGHT; ++h)
        for (uint32_t w = 0; w < WIDTH; ++w)
        {
            FreeImage_GetPixelColor(image, (unsigned int)w, HEIGHT - (unsigned int)h - 1, &color);
            int r = color.rgbRed, g = color.rgbGreen, b = color.rgbBlue;

            if (grayScale)
                Set((r * 0.3f + g * 0.59f + b * 0.11f) / (normalize ? 255.0f : 1.f), w, h);
            else
            {
                Set(r / (normalize ? 255.0f : 1.f), w, h, 0);
                Set(g / (normalize ? 255.0f : 1.f), w, h, 1);
                Set(b / (normalize ? 255.0f : 1.f), w, h, 2);
            }
        }

        FreeImage_Unload(image);
	}

    //////////////////////////////////////////////////////////////////////////
    void Tensor::SaveAsImage(const string& imageFile, bool denormalize) const
    {
        ImageLibInit();

        auto format = GetFormat(imageFile);
        assert(format != FIF_UNKNOWN);

        const uint32_t TENSOR_WIDTH = Width();
        const uint32_t TENSOR_HEIGHT = Height();
        const uint32_t IMG_ROWS = (uint32_t)ceil(::sqrt((float)Batch()));
        const uint32_t IMG_COLS = (uint32_t)ceil(::sqrt((float)Batch()));
        const uint32_t IMG_WIDTH = IMG_ROWS * TENSOR_WIDTH;
        const uint32_t IMG_HEIGHT = IMG_COLS * TENSOR_HEIGHT;
        const bool GRAYSCALE = (Depth() == 1);
        
        RGBQUAD color;
        color.rgbRed = color.rgbGreen = color.rgbBlue = 255;
        
        FIBITMAP* image = FreeImage_Allocate(IMG_WIDTH, IMG_HEIGHT, 24);
        FreeImage_FillBackground(image, &color);

        for (uint32_t n = 0; n < Batch(); ++n)
        for (uint32_t h = 0; h < Height(); ++h)
        for (uint32_t w = 0; w < Width(); ++w)
        {
            color.rgbRed = (int)(Get(w, h, 0, n) * (denormalize ? 255.0f : 1.f));
            color.rgbGreen = GRAYSCALE ? color.rgbRed : (int)(Get(w, h, 1, n) * (denormalize ? 255.0f : 1.f));
            color.rgbBlue = GRAYSCALE ? color.rgbRed : (int)(Get(w, h, 2, n) * (denormalize ? 255.0f : 1.f));

            FreeImage_SetPixelColor(image, (n % IMG_COLS) * TENSOR_WIDTH + w, IMG_HEIGHT - ((n / IMG_COLS) * TENSOR_HEIGHT + h) - 1, &color);
        }

        FreeImage_Save(format, image, imageFile.c_str());
        FreeImage_Unload(image);
    }

	//////////////////////////////////////////////////////////////////////////
	void Tensor::SetDefaultOpMode(EOpMode mode)
	{
		g_DefaultOp = GetOpFromMode(mode);
	}

    //////////////////////////////////////////////////////////////////////////
    void Tensor::SetForcedOpMode(EOpMode mode)
    {
        g_ForcedOp = GetOpFromMode(mode);
    }

    //////////////////////////////////////////////////////////////////////////
    void Tensor::ClearForcedOpMode()
    {
        g_ForcedOp = nullptr;
    }

	//////////////////////////////////////////////////////////////////////////
	void Tensor::SetOpMode(EOpMode mode)
	{
		m_Op = GetOpFromMode(mode);
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
    Tensor& Tensor::FillWithRand(int seed, float min, float max, uint32_t offset)
	{
		OverrideHost();

		auto fillUp = [&](Random& rng)
		{
			for (uint32_t i = offset; i < m_Values.size(); ++i)
				m_Values[i] = min + (max - min) * rng.NextFloat();
		};

        if (seed > 0)
        {
            Random tmpRng(seed);
            fillUp(tmpRng);
        }
        else
            fillUp(GlobalRng());

		return *this;
	}

	//////////////////////////////////////////////////////////////////////////
    Tensor& Tensor::FillWithRange(float start, float increment, uint32_t offset)
	{
		OverrideHost();
        for (uint32_t i = offset; i < m_Values.size(); ++i)
			m_Values[i] = start + i * increment;
		return *this;
	}

	//////////////////////////////////////////////////////////////////////////
	Tensor& Tensor::FillWithValue(float value, uint32_t offset)
	{
		OverrideHost();
		for (uint32_t i = offset; i < m_Values.size(); ++i)
			m_Values[i] = value;
		return *this;
	}

    //////////////////////////////////////////////////////////////////////////
    Tensor& Tensor::FillWithFunc(const function<float()>& func, uint32_t offset)
    {
        OverrideHost();
        for (uint32_t i = offset; i < m_Values.size(); ++i)
            m_Values[i] = func();
        return *this;
    }

    //////////////////////////////////////////////////////////////////////////
	void Tensor::Zero()
	{
        if (m_CurrentLocation == ELocation::Host)
            fill(m_Values.begin(), m_Values.end(), 0.f);
        else
            GetDeviceVar().ZeroOnDevice();
	}

    //////////////////////////////////////////////////////////////////////////
    Tensor Tensor::ToNCHW() const
    {
        Tensor result(GetShape());

        uint32_t i = 0;
        for (uint32_t n = 0; n < Batch(); ++n)
        for (uint32_t h = 0; h < Height(); ++h)
        for (uint32_t w = 0; w < Width(); ++w)
        {
            for (uint32_t j = 0; j < Depth(); ++j, ++i)
            {
                result(w, h, j, n) = m_Values[i];
            }
        }

        return result;
    }

    //////////////////////////////////////////////////////////////////////////
    Tensor Tensor::ToNHWC() const
    {
        Tensor result(GetShape());

        uint32_t i = 0;
        for (uint32_t n = 0; n < Batch(); ++n)
        for (uint32_t h = 0; h < Height(); ++h)
        for (uint32_t w = 0; w < Width(); ++w)
        {
            for (uint32_t j = 0; j < Depth(); ++j, ++i)
            {
                result.m_Values[i] = Get(w, h, j, n);
            }
        }

        return result;
    }

    //////////////////////////////////////////////////////////////////////////
	void Tensor::Mul(bool transposeT, const Tensor& t, Tensor& result) const
	{
		assert((!transposeT && Width() == t.Height()) || (transposeT && Width() == t.Width()));
		assert(t.Depth() == Depth());

		Op()->Mul(false, transposeT, *this, t, result);
	}

	//////////////////////////////////////////////////////////////////////////
	Tensor Tensor::Mul(bool transposeT, const Tensor& t) const
	{
		Tensor result(Shape(transposeT ? t.m_Shape.Height() : t.m_Shape.Width(), Height(), Depth(), max(Batch(), t.Batch())));
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
		Op()->MulElem(*this, t, result);
	}


	//////////////////////////////////////////////////////////////////////////
	Tensor Tensor::MulElem(const Tensor& t) const
	{
        Tensor result(Shape(max(Width(), t.Width()), max(Height(), t.Height()), max(Depth(), t.Depth()), max(Batch(), t.Batch())));
		MulElem(t, result);
		return result;
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::Mul(float v, Tensor& result) const
	{
		CopyToHost();
		result.OverrideHost();

		for (uint32_t i = 0; i < m_Values.size(); ++i)
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

		for (uint32_t i = 0; i < m_Values.size(); ++i)
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
        Mul(1 / v, result);
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
		Op()->Add(alpha, *this, beta, t, result);
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::Add(const Tensor& t, Tensor& result) const
	{
		Add(1, 1, t, result);
	}

	//////////////////////////////////////////////////////////////////////////
	Tensor Tensor::Add(const Tensor& t) const
	{
        Tensor result(Shape(max(Width(), t.Width()), max(Height(), t.Height()), max(Depth(), t.Depth()), max(Batch(), t.Batch())));
		Add(t, result);
		return result;
	}

	//////////////////////////////////////////////////////////////////////////
	Tensor Tensor::Add(float alpha, float beta, const Tensor& t) const
	{
        Tensor result(Shape(max(Width(), t.Width()), max(Height(), t.Height()), max(Depth(), t.Depth()), max(Batch(), t.Batch())));
		Add(alpha, beta, t, result);
		return result;
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::Add(float v, Tensor& result) const
	{
		CopyToHost();
		for (uint32_t i = 0; i < m_Values.size(); ++i)
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
		Op()->Sub(*this, t, result);
	}

	//////////////////////////////////////////////////////////////////////////
	Tensor Tensor::Sub(const Tensor& t) const
	{
		Tensor result(Shape(max(Width(), t.Width()), max(Height(), t.Height()), max(Depth(), t.Depth()), max(Batch(), t.Batch())));
		Sub(t, result);
		return result;
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::Sub(float v, Tensor& result) const
	{
		CopyToHost();
		for (uint32_t i = 0; i < m_Values.size(); ++i)
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
		for (uint32_t i = 0; i < m_Values.size(); ++i)
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

        uint32_t batchLen = BatchLength();

		for (uint32_t b = 0; b < Batch(); ++b)
		for (uint32_t i = 0; i < batchLen; ++i)
			result(i, i, 0, b) = m_Values[b * batchLen + i];

		return result;
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::Map(const function<float(float)>& func, Tensor& result) const
	{
		Op()->Map(func, *this, result);
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
		Op()->Map(func, *this, other, result);
	}

	//////////////////////////////////////////////////////////////////////////
	Tensor Tensor::Map(const function<float(float, float)>& func, const Tensor& other) const
	{
		Tensor result(Shape(max(Width(), other.Width()), max(Height(), other.Height()), max(Depth(), other.Depth()), max(Batch(), other.Batch())));
		Map(func, other, result);
		return result;
	}

    //////////////////////////////////////////////////////////////////////////
    template <int W, int H, int D, int N>
    Tensor SumTemplate(const Tensor& input, EAxis axis)
    {
        Tensor sum(Shape(W ? 1 : input.Width(), H ? 1 : input.Height(), D ? 1 : input.Depth(), N ? 1 : input.Batch()));
        input.Sum(axis, sum);
        return sum;
    }

	//////////////////////////////////////////////////////////////////////////
    Tensor Tensor::Sum(EAxis axis) const
	{
        if (axis == GlobalAxis)
            return SumTemplate<1, 1, 1, 1>(*this, axis);
        if (axis == WidthAxis)
            return SumTemplate<1, 0, 0, 0>(*this, axis);
        if (axis == HeightAxis)
            return SumTemplate<0, 1, 0, 0>(*this, axis);
        if (axis == DepthAxis)
            return SumTemplate<0, 0, 1, 0>(*this, axis);
        if (axis == BatchAxis)
            return SumTemplate<0, 0, 0, 1>(*this, axis);
        if (axis == _012Axes)
            return SumTemplate<1, 1, 1, 0>(*this, axis);
        if (axis == _013Axes)
            return SumTemplate<1, 1, 0, 1>(*this, axis);
        if (axis == _123Axes)
            return SumTemplate<0, 1, 1, 1>(*this, axis);

        assert(false);
        return Tensor();
	}

    //////////////////////////////////////////////////////////////////////////
    void Tensor::Sum(EAxis axis, Tensor& output) const
    {
        Op()->Sum(*this, axis, output);
    }

    //////////////////////////////////////////////////////////////////////////
    template <int W, int H, int D, int N>
    Tensor MeanTemplate(const Tensor& input, EAxis axis)
    {
        Tensor mean(Shape(W ? 1 : input.Width(), H ? 1 : input.Height(), D ? 1 : input.Depth(), N ? 1 : input.Batch()));
        input.Mean(axis, mean);
        return mean;
    }

    //////////////////////////////////////////////////////////////////////////
    Tensor Tensor::Mean(EAxis axis) const
	{
        if (axis == GlobalAxis)
            return MeanTemplate<1, 1, 1, 1>(*this, axis);
        if (axis == WidthAxis)
            return MeanTemplate<1, 0, 0, 0>(*this, axis);
        if (axis == HeightAxis)
            return MeanTemplate<0, 1, 0, 0>(*this, axis);
        if (axis == DepthAxis)
            return MeanTemplate<0, 0, 1, 0>(*this, axis);
        if (axis == BatchAxis)
            return MeanTemplate<0, 0, 0, 1>(*this, axis);
        if (axis == _012Axes)
            return MeanTemplate<1, 1, 1, 0>(*this, axis);
        if (axis == _013Axes)
            return MeanTemplate<1, 1, 0, 1>(*this, axis);
        if (axis == _123Axes)
            return MeanTemplate<0, 1, 1, 1>(*this, axis);

        assert(false);
        return Tensor();
	}

    //////////////////////////////////////////////////////////////////////////
    void Tensor::Mean(EAxis axis, Tensor& output) const
    {
        Sum(axis, output);

        if (axis == GlobalAxis)
            output.Div((float)Length(), output);
        if (axis == WidthAxis)
            output.Div((float)Width(), output);
        if (axis == HeightAxis)
            output.Div((float)Height(), output);
        if (axis == DepthAxis)
            output.Div((float)Depth(), output);
        if (axis == BatchAxis)
            output.Div((float)Batch(), output);
        if (axis == _012Axes)
            output.Div((float)(Len(0)*Len(1)*Len(2)), output);
        if (axis == _013Axes)
            output.Div((float)(Len(0)*Len(1)*Len(3)), output);
        if (axis == _123Axes)
            output.Div((float)(Len(1)*Len(2)*Len(3)), output);
    }

	//////////////////////////////////////////////////////////////////////////
	Tensor Tensor::MergeIntoBatch(const vector<Tensor>& tensors)
	{
		/*if (tensors.Count == 0)
			throw new Exception("List cannot be empty.");*/

		Tensor output(Shape(tensors[0].Width(), tensors[0].Height(), tensors[0].Depth(), (int)tensors.size()));

		for (uint32_t n = 0; n < tensors.size(); ++n)
		{
			const Tensor& t = tensors[n];
			t.CopyToHost();
			copy(t.m_Values.begin(), t.m_Values.end(), output.m_Values.begin() + t.Length() * n);
		}

		return output;
	}

	//////////////////////////////////////////////////////////////////////////
	Tensor Tensor::MergeIntoDepth(const vector<Tensor>& tensors, uint32_t forcedDepth)
	{
		/*if (tensors.Count == 0)
			throw new Exception("List cannot be empty.");*/

		Tensor output(Shape(tensors[0].Width(), tensors[0].Height(), max((uint32_t)tensors.size(), forcedDepth)));

		const Tensor& t = tensors[0];
		t.CopyToHost();

		uint32_t t0_copies = forcedDepth > 0 ? forcedDepth - (uint32_t)tensors.size() : 0;

		for (uint32_t n = 0; n < t0_copies; ++n)
		{
			copy(t.m_Values.begin(), t.m_Values.end(), output.m_Values.begin() + t.Length() * n);
		}

		for (uint32_t n = t0_copies; n < output.Depth(); ++n)
		{
			const Tensor& t = tensors[n - t0_copies];
			t.CopyToHost();
			copy(t.m_Values.begin(), t.m_Values.end(), output.m_Values.begin() + t.Length() * n);
		}

		return output;
	}

    //////////////////////////////////////////////////////////////////////////
    template <int W, int H, int D, int N>
    void ConcatTemplate(const const_tensor_ptr_vec_t& inputs, Tensor& output)
    {
        auto& shape = inputs[0]->GetShape();
        const uint32_t width = shape.Width();
        const uint32_t height = shape.Height();
        const uint32_t depth = shape.Depth();
        const uint32_t batch = shape.Batch();

        for (uint32_t i = 0; i < (uint32_t)inputs.size(); ++i)
        {
            auto& inputValues = inputs[0]->GetValues();
            size_t j = 0;
            for (uint32_t n = 0; n < batch; ++n)
            for (uint32_t d = 0; d < depth; ++d)
            for (uint32_t h = 0; h < height; ++h)
            for (uint32_t w = 0; w < width; ++w, ++j)
                output(w + (W ? width * i : 0), h + (H ? height * i : 0), d + (D ? depth * i : 0), n + (N ? batch * i : 0)) = inputValues[j];
        }
    }

	//////////////////////////////////////////////////////////////////////////
	void Tensor::Concat(EAxis axis, const const_tensor_ptr_vec_t& inputs, Tensor& result)
	{
        result.OverrideHost();

        if (axis == BatchAxis)
        {
            uint32_t elementsCopied = 0;
            for (uint32_t i = 0; i < inputs.size(); ++i)
            {
                inputs[i]->CopyToHost();
                copy(inputs[i]->m_Values.begin(), inputs[i]->m_Values.end(), result.m_Values.begin() + elementsCopied);
                elementsCopied += inputs[i]->Length();
            }
        }
        else if (axis == WidthAxis)
        {
            ConcatTemplate<1, 0, 0, 0>(inputs, result);
        }
        else if (axis == HeightAxis)
        {
            ConcatTemplate<0, 1, 0, 0>(inputs, result);
        }
        else if (axis == DepthAxis)
        {
            ConcatTemplate<0, 0, 1, 0>(inputs, result);
        }
        else
            assert(false); // not supported
	}

    //////////////////////////////////////////////////////////////////////////
    template <int W, int H, int D, int N>
    void SplitTemplate(const Tensor& input, tensor_ptr_vec_t& outputs)
    {
        auto& shape = outputs[0]->GetShape();
        const uint32_t width = shape.Width();
        const uint32_t height = shape.Height();
        const uint32_t depth = shape.Depth();
        const uint32_t batch = shape.Batch();

        for (uint32_t i = 0; i < (uint32_t)outputs.size(); ++i)
        {
            auto& outputValues = outputs[0]->GetValues();
            size_t j = 0;
            for (uint32_t n = 0; n < batch; ++n)
            for (uint32_t d = 0; d < depth; ++d)
            for (uint32_t h = 0; h < height; ++h)
            for (uint32_t w = 0; w < width; ++w, ++j)
                outputValues[j] = input(w + (W ? width * i : 0), h + (H ? height * i : 0), d + (D ? depth * i : 0), n + (N ? batch * i : 0));
        }
    }

	//////////////////////////////////////////////////////////////////////////
	void Tensor::Split(EAxis axis, tensor_ptr_vec_t& outputs) const
	{
		CopyToHost();

        if (axis == BatchAxis)
        {
            uint32_t elementsCopied = 0;
            uint32_t singleOutputLen = Length() / (uint32_t)outputs.size();

            for (uint32_t i = 0; i < outputs.size(); ++i)
            {
                outputs[i]->OverrideHost();
                copy(m_Values.begin() + elementsCopied, m_Values.begin() + elementsCopied + singleOutputLen, outputs[i]->m_Values.begin());
                elementsCopied += singleOutputLen;
            }
        }
        else if (axis == WidthAxis)
        {
            SplitTemplate<1, 0, 0, 0>(*this, outputs);
        }
        else if (axis == HeightAxis)
        {
            SplitTemplate<0, 1, 0, 0>(*this, outputs);
        }
        else if (axis == DepthAxis)
        {
            SplitTemplate<0, 0, 1, 0>(*this, outputs);
        }
        else
            assert(false); // not supported yet
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::MergeMin(const const_tensor_ptr_vec_t& inputs, Tensor& result)
	{
		inputs[0]->CopyTo(result);
		for (uint32_t i = 1; i < inputs.size(); ++i)
		for (uint32_t j = 0; j < result.Length(); ++j)
			result.m_Values[j] = result.m_Values[j] > inputs[i]->m_Values[j] ? inputs[i]->m_Values[j] : result.m_Values[j];
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::MergeMax(const const_tensor_ptr_vec_t& inputs, Tensor& result)
	{
		inputs[0]->CopyTo(result);
		for (uint32_t i = 1; i < inputs.size(); ++i)
		for (uint32_t j = 0; j < result.Length(); ++j)
			result.m_Values[j] = result.m_Values[j] < inputs[i]->m_Values[j] ? inputs[i]->m_Values[j] : result.m_Values[j];
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::MergeSum(const const_tensor_ptr_vec_t& inputs, Tensor& result)
	{
        result.OverrideHost();
		result.Zero();
		for (uint32_t i = 0; i < inputs.size(); ++i)
		for (uint32_t j = 0; j < result.Length(); ++j)
			result.m_Values[j] += inputs[i]->m_Values[j];
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::MergeAvg(const const_tensor_ptr_vec_t& inputs, Tensor& result)
	{
		MergeSum(inputs, result);
		result.Div((float)inputs.size(), result);
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::MergeMinMaxGradient(const Tensor& output, const const_tensor_ptr_vec_t& inputs, const Tensor& outputGradient, tensor_ptr_vec_t& results)
	{
		for (uint32_t i = 0; i < inputs.size(); ++i)
		{
            results[i]->OverrideHost();
			results[i]->Zero();
			for (uint32_t j = 0; j < output.Length(); ++j)
				results[i]->m_Values[j] = inputs[i]->m_Values[j] == output.m_Values[j] ? outputGradient.m_Values[j] : 0;
		}
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::MergeSumGradient(const Tensor& output, const const_tensor_ptr_vec_t& inputs, const Tensor& outputGradient, tensor_ptr_vec_t& results)
	{
		for (uint32_t i = 0; i < inputs.size(); ++i)
			outputGradient.CopyTo(*results[i]);
	}

	//////////////////////////////////////////////////////////////////////////
    void Tensor::MergeAvgGradient(const Tensor& output, const const_tensor_ptr_vec_t& inputs, const Tensor& outputGradient, tensor_ptr_vec_t& results)
	{
		MergeSumGradient(output, inputs, outputGradient, results);
		for (uint32_t i = 0; i < results.size(); ++i)
			results[i]->Div((float)results.size(), *results[i]);
	}

    //////////////////////////////////////////////////////////////////////////
    vector<EAxis> Tensor::FillUpTranposeAxis(const vector<EAxis>& axes)
    {
        vector<EAxis> permutation(axes);
        // add unlisted axis at the end in order of axis defining shape
        for (int a = WidthAxis; a <= BatchAxis; ++a)
        {
            if (find(axes.begin(), axes.end(), a) == axes.end())
                permutation.push_back((EAxis)a);
        }

        return permutation;
    }

    //////////////////////////////////////////////////////////////////////////
    Tensor Tensor::Normalized(EAxis axis, Tensor& result, ENormMode normMode, Tensor* savedNorm) const
    {
        CopyToHost();
        result.OverrideHost();

        assert(result.GetShape() == GetShape());
            
        /*if (axis == Sample)
        {
            Tensor norm;
            
            if (savedNorm)
                norm = *savedNorm;
            else
            {
                norm = Tensor(Shape(Batch()));
                norm.FillWithValue(0);

                for (uint32_t n = 0; n < Batch(); ++n)
                for (uint32_t i = 0, idx = n * BatchLength(); i < BatchLength(); ++i, ++idx)
                    norm(n) += normMode == ENormMode::L1 ? abs(m_Values[idx]) : (m_Values[idx] * m_Values[idx]);

                if (normMode == ENormMode::L2)
                {
                    for (uint32_t i = 0; i < norm.Length(); ++i)
                        norm.m_Values[i] = sqrt(norm.m_Values[i]);
                }
            }
        
            for (uint32_t n = 0; n < Batch(); ++n)
            for (uint32_t i = 0, idx = n * BatchLength(); i < BatchLength(); ++i, ++idx)
                result.m_Values[idx] = m_Values[idx] / norm(n);

            return norm;
        }
        else */if (axis == BatchAxis)
        {
            Tensor norm;

            if (savedNorm)
                norm = *savedNorm;
            else
            {
                norm = Tensor(Shape(Width(), Height(), Depth(), 1));
                norm.FillWithValue(0);

                for (uint32_t n = 0; n < Batch(); ++n)
                for (uint32_t d = 0; d < Depth(); ++d)
                for (uint32_t h = 0; h < Height(); ++h)
                for (uint32_t w = 0; w < Width(); ++w)
                    norm(w, h, d) += normMode == ENormMode::L1 ? abs(Get(w, h, d, n)) : (Get(w, h, d, n) * Get(w, h, d, n));

                if (normMode == ENormMode::L2)
                {
                    for (uint32_t i = 0; i < norm.Length(); ++i)
                        norm.m_Values[i] = ::sqrt(norm.m_Values[i]);
                }
            }

            for (uint32_t n = 0; n < Batch(); ++n)
            for (uint32_t d = 0; d < Depth(); ++d)
            for (uint32_t h = 0; h < Height(); ++h)
            for (uint32_t w = 0; w < Width(); ++w)
                result(w, h, d, n) = Get(w, h, d, n) / norm(w, h, d);

            return norm;
        }
        else if (axis == GlobalAxis)
        {
            Tensor norm;

            if (savedNorm)
                norm = *savedNorm;
            else
            {
                norm = Tensor({ 0 }, Shape(1));
                for (uint32_t i = 0; i < Length(); ++i)
                    norm(0) += normMode == ENormMode::L1 ? abs(m_Values[i]) : (m_Values[i] * m_Values[i]);

                if (normMode == ENormMode::L2)
                {
                    for (uint32_t i = 0; i < norm.Length(); ++i)
                        norm.m_Values[i] = ::sqrt(norm.m_Values[i]);
                }
            }

            for (uint32_t i = 0; i < Length(); ++i)
                result.m_Values[i] = m_Values[i] / norm(0);

            return norm;
        }
        else
        {
            assert(false && "Axis not supported!");
            return Tensor();
        }
    }

    //////////////////////////////////////////////////////////////////////////
    Tensor Tensor::Normalized(EAxis axis, ENormMode normMode, Tensor* savedNorm) const
    {
        Tensor result(GetShape());
        Normalized(axis, result, normMode, savedNorm);
        return result;
    }

    //////////////////////////////////////////////////////////////////////////
    pair<Tensor, Tensor> Tensor::NormalizedMinMax(EAxis axis, Tensor& result, float scaleMin, float scaleMax, Tensor* savedMin, Tensor* savedMax) const
    {
        CopyToHost();
        result.OverrideHost();

        assert(result.GetShape() == GetShape());
            
        const float rangeSpread = scaleMax - scaleMin;

        Tensor min = savedMin ? *savedMin : Min(axis);
        Tensor max = savedMax ? *savedMax : Max(axis);

        /*if (axis == Sample)
        {
            assert(min.Width() == Batch());
            assert(max.Width() == Batch());

            for (uint32_t n = 0; n < Batch(); ++n)
            {
                const float minVal = min(n);
                const float spread = max(n) - minVal;

                for (uint32_t i = 0, idx = n * BatchLength(); i < BatchLength(); ++i, ++idx)
                    result.m_Values[idx] = rangeSpread * (m_Values[idx] - minVal) / spread + scaleMin;
            }
        }
        else */if (axis == BatchAxis)
        {
            assert(SameDimensionsExceptBatches(min) && min.Batch() == 1);
            assert(SameDimensionsExceptBatches(max) && max.Batch() == 1);

            for (uint32_t d = 0; d < Depth(); ++d)
            for (uint32_t h = 0; h < Height(); ++h)
            for (uint32_t w = 0; w < Width(); ++w)
            {
                const float minVal = min(w, h, d);
                const float spread = max(w, h, d) - minVal;

                for (uint32_t n = 0; n < Batch(); ++n)
                    result(w, h, d, n) = rangeSpread * (Get(w, h, d, n) - minVal) / spread + scaleMin;
            }
        }
        else if (axis == GlobalAxis)
        {
            const float minVal = min(0);
            const float spread = max(0) - minVal;

            for (uint32_t i = 0; i < Length(); ++i)
                result.m_Values[i] = rangeSpread * (m_Values[i] - minVal) / spread + scaleMin;
        }
        else
        {
            assert(false && "Axis not supported!");
            return make_pair(Tensor(), Tensor());
        }

        return make_pair(min, max);
    }

    //////////////////////////////////////////////////////////////////////////
    Tensor Tensor::NormalizedMinMax(EAxis axis, float scaleMin, float scaleMax, Tensor* savedMin, Tensor* savedMax) const
    {
        Tensor result(GetShape());
        NormalizedMinMax(axis, result, scaleMin, scaleMax, savedMin, savedMax);
        return result;
    }

    //////////////////////////////////////////////////////////////////////////
    pair<Tensor, Tensor> Tensor::Standardized(EAxis axis, Tensor& result, Tensor* mean, Tensor* invVariance) const
    {
        if (axis == BatchAxis)
        {
            if (mean && invVariance)
            {
                Tensor xmu = Sub(*mean);
                xmu.MulElem(*invVariance, result);
                return make_pair(*mean, *invVariance);
            }

            float n = (float)Batch();
            Tensor xmean = Mean(BatchAxis);
            Tensor xmu = Sub(xmean);
            Tensor variance = xmu.Map([](float x) { return x * x; }).Sum(BatchAxis).Mul(1.f / n);
            Tensor invVar = variance.Map([](float x) { return 1.f / x; });
            xmu.MulElem(invVar, result);
            return make_pair(xmean, invVar);
        }
        else
        {
            assert(false && "Axis not supported!");
            return make_pair(Tensor(), Tensor());
        }
    }

    //////////////////////////////////////////////////////////////////////////
    Tensor Tensor::Standardized(EAxis axis, Tensor* mean, Tensor* invVariance) const
    {
        Tensor result(GetShape());
        Standardized(axis, result, mean, invVariance);
        return result;
    }

    //////////////////////////////////////////////////////////////////////////
    Tensor Tensor::ArgMax(EAxis axis) const
	{
        Tensor maxIndex(Shape(1));
        if (axis == GlobalAxis)
            maxIndex.Resize(Shape(1, 1, 1, 1));
        else if (axis == WidthAxis)
            maxIndex.Resize(Shape(1, Height(), Depth(), Batch()));
        else if (axis == HeightAxis)
            maxIndex.Resize(Shape(Width(), 1, Depth(), Batch()));
        else if (axis == DepthAxis)
            maxIndex.Resize(Shape(Width(), Height(), 1, Batch()));
        else if (axis == BatchAxis)
            maxIndex.Resize(Shape(Width(), Height(), Depth(), 1));
        else if (axis == _012Axes)
            maxIndex.Resize(Shape(1, 1, 1, Len(3)));
        else if (axis == _013Axes)
            maxIndex.Resize(Shape(1, 1, Len(2), 1));
        else if (axis == _123Axes)
            maxIndex.Resize(Shape(Len(0), 1, 1, 1));
            
		Max(axis, &maxIndex);
		return maxIndex;
	}

    //////////////////////////////////////////////////////////////////////////
    Tensor Tensor::ArgMin(EAxis axis) const
    {
        Tensor minIndex(Shape(1));
        if (axis == GlobalAxis)
            minIndex.Resize(Shape(1, 1, 1, 1));
        else if (axis == WidthAxis)
            minIndex.Resize(Shape(1, Height(), Depth(), Batch()));
        else if (axis == HeightAxis)
            minIndex.Resize(Shape(Width(), 1, Depth(), Batch()));
        else if (axis == DepthAxis)
            minIndex.Resize(Shape(Width(), Height(), 1, Batch()));
        else if (axis == BatchAxis)
            minIndex.Resize(Shape(Width(), Height(), Depth(), 1));
        else if (axis == _012Axes)
            minIndex.Resize(Shape(1, 1, 1, Len(3)));
        else if (axis == _013Axes)
            minIndex.Resize(Shape(1, 1, Len(2), 1));
        else if (axis == _123Axes)
            minIndex.Resize(Shape(Len(0), 1, 1, 1));

        Min(axis, &minIndex);
        return minIndex;
    }

	//////////////////////////////////////////////////////////////////////////
	Tensor Tensor::Transposed() const
	{
		Tensor result(Shape(Height(), Width(), Depth(), Batch()));
		Transpose(result);
		return result;
	}

    //////////////////////////////////////////////////////////////////////////
    Tensor Tensor::Transposed(const vector<EAxis>& axes) const
    {
        Tensor result;
        Transpose(axes, result);
        return result;
    }

    //////////////////////////////////////////////////////////////////////////
    void Tensor::Transpose(const vector<EAxis>& axes, Tensor& result) const
    {
        vector<EAxis> permutation = FillUpTranposeAxis(axes);

        result.Resize(Shape(m_Shape.Dimensions[permutation[0]], m_Shape.Dimensions[permutation[1]], m_Shape.Dimensions[permutation[2]], m_Shape.Dimensions[permutation[3]]));

        for (uint32_t n = 0; n < result.Batch(); ++n)
        for (uint32_t d = 0; d < result.Depth(); ++d)
        for (uint32_t h = 0; h < result.Height(); ++h)
        for (uint32_t w = 0; w < result.Width(); ++w)
        {
            int inIndex = w * m_Shape.Stride[permutation[0]] + h * m_Shape.Stride[permutation[1]] + d * m_Shape.Stride[permutation[2]] + n * m_Shape.Stride[permutation[3]];
            result(w, h, d, n) = m_Values[inIndex];
        }
    }

    //////////////////////////////////////////////////////////////////////////
	void Tensor::Transpose(Tensor& result) const
	{
		Op()->Transpose(*this, result);
	}

    //////////////////////////////////////////////////////////////////////////
	Tensor Tensor::Reshaped(const Shape& shape) const
	{
		return Tensor(m_Values, m_Shape.Reshaped((int)shape.Width(), (int)shape.Height(), (int)shape.Depth(), (int)shape.Batch()));
	}

    //////////////////////////////////////////////////////////////////////////
    void Tensor::Reshaped(const Shape& shape, Tensor& output) const
    {
        CopyTo(output);
        output.Reshaped(shape);
    }

	//////////////////////////////////////////////////////////////////////////
	void Tensor::Reshape(const Shape& shape)
	{
		m_Shape = m_Shape.Reshaped((int)shape.Width(), (int)shape.Height(), (int)shape.Depth(), (int)shape.Batch());
	}

    //////////////////////////////////////////////////////////////////////////
    void Tensor::Resize(const Shape& shape)
    {
        if (m_Shape == shape)
            return;

        m_Shape = shape;
        m_Values.resize(shape.Length);
        m_GpuData.Release();
    }

	//////////////////////////////////////////////////////////////////////////
	Tensor Tensor::Resized(uint32_t width, uint32_t height, uint32_t depth) const
	{
		uint32_t newBatchLength = width * height * depth;
		Tensor result(Shape(width, height, depth, m_Shape.Batch()));
		for (uint32_t n = 0; n < Batch(); ++n)
		for (uint32_t i = 0, idx = n * newBatchLength; i < newBatchLength; ++i, ++idx)
			result.m_Values[idx] = m_Values[n * BatchLength() + i % BatchLength()];
		return result;
	}

	//////////////////////////////////////////////////////////////////////////
	Tensor Tensor::FlattenHoriz() const
	{
        return Reshaped(m_Shape.Reshaped(Shape::Auto, 1, 1, (int)m_Shape.Batch()));
	}

	//////////////////////////////////////////////////////////////////////////
	Tensor Tensor::FlattenVert() const
	{
		return Reshaped(m_Shape.Reshaped(1, Shape::Auto, 1, (int)m_Shape.Batch()));
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::Rotated180(Tensor& result) const
	{
		assert(SameDimensionsExceptBatches(result));

		for (uint32_t n = 0; n < Batch(); ++n)
		for (uint32_t d = 0; d < Depth(); ++d)
		for (int h = Height() - 1; h >= 0; --h)
		for (int w = Width() - 1; w >= 0; --w)
			result.Set(Get(Width() - (uint32_t)w - 1, Height() - (uint32_t)h - 1, d, n), w, h, d, n);
	}

	//////////////////////////////////////////////////////////////////////////
	Tensor Tensor::Rotated180() const
	{
		Tensor result(m_Shape);
		Rotated180(result);
		return result;
	}

    //////////////////////////////////////////////////////////////////////////
    void Tensor::Conv2D(const Tensor& kernels, uint32_t stride, uint32_t padding, EDataFormat dataFormat, Tensor& result) const
	{
        assert((dataFormat == NCHW ? Depth() : Len(0)) == kernels.Depth());
		Op()->Conv2D(*this, kernels, stride, padding, padding, dataFormat, result);
	}

	//////////////////////////////////////////////////////////////////////////
    Tensor Tensor::Conv2D(const Tensor& kernels, uint32_t stride, uint32_t padding, EDataFormat dataFormat) const
	{
		Tensor result(GetConvOutputShape(GetShape(), kernels.Batch(), kernels.Width(), kernels.Height(), stride, padding, padding, dataFormat));
		Conv2D(kernels, stride, padding, dataFormat, result);
		return result;
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::Conv2DInputsGradient(const Tensor& gradient, const Tensor& kernels, uint32_t stride, uint32_t padding, EDataFormat dataFormat, Tensor& inputsGradient) const
	{
		inputsGradient.Zero();
		Op()->Conv2DInputGradient(gradient, kernels, stride, padding, padding, dataFormat, inputsGradient);
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::Conv2DKernelsGradient(const Tensor& input, const Tensor& gradient, uint32_t stride, uint32_t padding, EDataFormat dataFormat, Tensor& kernelsGradient) const
	{
		kernelsGradient.Zero();
		Op()->Conv2DKernelsGradient(input, gradient, stride, padding, padding, dataFormat, kernelsGradient);
	}

    //////////////////////////////////////////////////////////////////////////
    void Tensor::Conv2DTransposed(const Tensor& kernels, uint32_t stride, uint32_t padding, EDataFormat dataFormat, Tensor& result) const
    {
        assert((dataFormat == NCHW ? Depth() : Len(0)) == kernels.Batch());
        Conv2DInputsGradient(*this, kernels, stride, padding, dataFormat, result);
    }

    //////////////////////////////////////////////////////////////////////////
    Tensor Tensor::Conv2DTransposed(const Tensor& kernels, uint32_t outputDepth, uint32_t stride, uint32_t padding, EDataFormat dataFormat) const
    {
        Tensor result(GetConvTransposeOutputShape(GetShape(), outputDepth, kernels.Width(), kernels.Height(), stride, padding, padding, dataFormat));
        Conv2DTransposed(kernels, stride, padding, dataFormat, result);
        return result;
    }

    //////////////////////////////////////////////////////////////////////////
    void Tensor::Conv2DTransposedInputsGradient(const Tensor& gradient, const Tensor& kernels, uint32_t stride, uint32_t padding, EDataFormat dataFormat, Tensor& inputsGradient) const
    {
        inputsGradient.Zero();
        gradient.Conv2D(kernels, stride, padding, dataFormat, inputsGradient);
    }

    //////////////////////////////////////////////////////////////////////////
    void Tensor::Conv2DTransposedKernelsGradient(const Tensor& input, const Tensor& gradient, uint32_t stride, uint32_t padding, EDataFormat dataFormat, Tensor& kernelsGradient) const
    {
        kernelsGradient.Zero();
        Op()->Conv2DKernelsGradient(gradient, input, stride, padding, padding, dataFormat, kernelsGradient);
    }

	//////////////////////////////////////////////////////////////////////////
	void Tensor::Pool2D(uint32_t filterSize, uint32_t stride, EPoolingMode type, uint32_t padding, EDataFormat dataFormat, Tensor& output) const
	{
		assert(output.Batch() == Batch());
        Op()->Pool2D(*this, filterSize, stride, type, padding, padding, dataFormat, output);
	}

	//////////////////////////////////////////////////////////////////////////
	Tensor Tensor::Pool2D(uint32_t filterSize, uint32_t stride, EPoolingMode type, uint32_t padding, EDataFormat dataFormat) const
	{
		Tensor result(GetConvOutputShape(GetShape(), GetShape().Depth(), filterSize, filterSize, stride, padding, padding, dataFormat));
		Pool2D(filterSize, stride, type, padding, dataFormat, result);

		return result;
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::Pool2DGradient(const Tensor& output, const Tensor& input, const Tensor& outputGradient, uint32_t filterSize, uint32_t stride, EPoolingMode type, uint32_t padding, EDataFormat dataFormat, Tensor& result) const
	{
		assert(output.SameDimensionsExceptBatches(outputGradient));
		Op()->Pool2DGradient(output, input, outputGradient, filterSize, stride, type, padding, padding, dataFormat, result);
	}

    //////////////////////////////////////////////////////////////////////////
    void Tensor::UpSample2D(uint32_t scaleFactor, Tensor& output) const
    {
        Op()->UpSample2D(*this, scaleFactor, output);
    }

    //////////////////////////////////////////////////////////////////////////
    Tensor Tensor::UpSample2D(uint32_t scaleFactor) const
    {
        Tensor result(Shape(Width() * scaleFactor, Height() * scaleFactor, Depth(), Batch()));
        UpSample2D(scaleFactor, result);
        return result;
    }

    //////////////////////////////////////////////////////////////////////////
    void Tensor::UpSample2DGradient(const Tensor& outputGradient, uint32_t scaleFactor, Tensor& inputGradient) const
    {
        Op()->UpSample2DGradient(outputGradient, scaleFactor, inputGradient);
    }

    //////////////////////////////////////////////////////////////////////////
    void Tensor::BatchNormalization(const Tensor& gamma, const Tensor& beta, float epsilon, const Tensor& runningMean, const Tensor& runningVar, Tensor& result) const
    {
        Op()->BatchNormalization(*this, m_Shape.Depth() > 1 ? Spatial : PerActivation, gamma, beta, epsilon, runningMean, runningVar, result);
    }

    //////////////////////////////////////////////////////////////////////////
    void Tensor::BatchNormalizationTrain(const Tensor& gamma, const Tensor& beta, float momentum, float epsilon, Tensor& runningMean, Tensor& runningVar, Tensor& saveMean, Tensor& saveInvVariance, Tensor& result) const
    {
        Op()->BatchNormalizationTrain(*this, m_Shape.Depth() > 1 ? Spatial : PerActivation, gamma, beta, momentum, epsilon, runningMean, runningVar, saveMean, saveInvVariance, result);
    }

    //////////////////////////////////////////////////////////////////////////
    void Tensor::BatchNormalizationGradient(const Tensor& input, const Tensor& gamma, float epsilon, const Tensor& outputGradient, const Tensor& savedMean, const Tensor& savedInvVariance, Tensor& gammaGradient, Tensor& betaGradient, bool trainable, Tensor& inputGradient) const
    {
        Op()->BatchNormalizationGradient(input, input.m_Shape.Depth() > 1 ? Spatial : PerActivation, gamma, epsilon, outputGradient, savedMean, savedInvVariance, gammaGradient, betaGradient, trainable, inputGradient);
    }

    //////////////////////////////////////////////////////////////////////////
    void Tensor::Dropout(float prob, Tensor& saveMask, Tensor& output) const
    {
        Op()->Dropout(*this, prob, saveMask, output);
    }

    //////////////////////////////////////////////////////////////////////////
    void Tensor::DropoutGradient(const Tensor& outputGradient, const Tensor& savedMask, Tensor& inputGradient) const
    {
        Op()->DropoutGradient(outputGradient, savedMask, inputGradient);
    }

    //////////////////////////////////////////////////////////////////////////
	string Tensor::ToString() const
	{
        return TensorFormatter::ToString(*this);
	}

	//////////////////////////////////////////////////////////////////////////
	bool Tensor::SameDimensionsExceptBatches(const Tensor& t) const
	{
		return Width() == t.Width() && Height() == t.Height() && Depth() == t.Depth();
	}

    //////////////////////////////////////////////////////////////////////////
    pair<uint32_t, uint32_t> Tensor::GetPadding(EPaddingMode paddingMode, uint32_t kernelWidth, uint32_t kernelHeight)
    {
        if (paddingMode == Valid)
            return make_pair(0, 0);

        if (paddingMode == Same)
            return make_pair((int)floor((float)kernelWidth / 2), (int)floor((float)kernelHeight / 2));

        if (paddingMode == Full)
            return make_pair(kernelWidth - 1, kernelHeight - 1);

        assert(false && "Unsupported padding mode!");
        return make_pair(0, 0);
    }

    //////////////////////////////////////////////////////////////////////////
    uint32_t Tensor::GetPadding(EPaddingMode paddingMode, uint32_t kernelSize)
    {
        return GetPadding(paddingMode, kernelSize, kernelSize).first;
    }

    //////////////////////////////////////////////////////////////////////////
    Neuro::Shape Tensor::GetPooling2DOutputShape(const Shape& inputShape, uint32_t kernelWidth, uint32_t kernelHeight, uint32_t stride, uint32_t paddingX, uint32_t paddingY, EDataFormat dataFormat)
    {
        assert(stride > 0);
        if (dataFormat == NCHW)
            return Shape((int)floor((inputShape.Width() + 2 * paddingX - kernelWidth) / (float)stride) + 1, 
                         (int)floor((inputShape.Height() + 2 * paddingY - kernelHeight) / (float)stride) + 1,
                         inputShape.Depth(),
                         inputShape.Batch());

        return Shape(inputShape.Len(0), 
                     (int)floor((inputShape.Len(1) + 2 * paddingX - kernelWidth) / (float)stride) + 1,
                     (int)floor((inputShape.Len(2) + 2 * paddingY - kernelHeight) / (float)stride) + 1,
                     inputShape.Len(3));
    }

    //////////////////////////////////////////////////////////////////////////
    Shape Tensor::GetConvOutputShape(const Shape& inputShape, uint32_t kernelsNum, uint32_t kernelWidth, uint32_t kernelHeight, uint32_t stride, uint32_t paddingX, uint32_t paddingY, EDataFormat dataFormat)
    {
        assert(stride > 0);
        if (dataFormat == NCHW)
            return Shape((int)floor((inputShape.Width() + 2 * paddingX - kernelWidth) / (float)stride) + 1, 
                         (int)floor((inputShape.Height() + 2 * paddingY - kernelHeight) / (float)stride) + 1,
                         kernelsNum,
                         inputShape.Batch());

        return Shape(kernelsNum, 
                     (int)floor((inputShape.Len(1) + 2 * paddingX - kernelWidth) / (float)stride) + 1,
                     (int)floor((inputShape.Len(2) + 2 * paddingY - kernelHeight) / (float)stride) + 1,
                     inputShape.Len(3));
    }

    //////////////////////////////////////////////////////////////////////////
    Shape Tensor::GetConvTransposeOutputShape(const Shape& inputShape, uint32_t outputDepth, uint32_t kernelWidth, uint32_t kernelHeight, uint32_t stride, uint32_t paddingX, uint32_t paddingY, EDataFormat dataFormat)
    {
        assert(stride > 0);
        if (dataFormat == NCHW)
            return Shape((inputShape.Width() - 1) * stride + kernelWidth - 2 * paddingX,
                         (inputShape.Height() - 1) * stride + kernelHeight - 2 * paddingY,
                         outputDepth, 
                         inputShape.Batch());

        return Shape(outputDepth, 
                     (inputShape.Len(1) - 1) * stride + kernelWidth - 2 * paddingX,
                     (inputShape.Len(2) - 1) * stride + kernelHeight - 2 * paddingY,
                     inputShape.Batch());
    }

    void Tensor::SaveBin(ostream& stream) const
    {
        m_Shape.SaveBin(stream);
        size_t valuesNum = m_Values.size();
        stream.write((const char*)&valuesNum, sizeof(valuesNum));
        stream.write((const char*)&m_Values[0], valuesNum * sizeof(float));
        size_t nameLen = m_Name.length();
        stream.write((const char*)&nameLen, sizeof(nameLen));
        stream.write(m_Name.c_str(), nameLen);
    }

    void Tensor::LoadBin(istream& stream)
    {
        m_Shape.LoadBin(stream);
        size_t valuesNum = 0;
        stream.read((char*)&valuesNum, sizeof(valuesNum));
        m_Values.resize(valuesNum);
        stream.read((char*)&m_Values[0], valuesNum * sizeof(float));
        size_t nameLen = 0;
        stream.read((char*)&nameLen, sizeof(nameLen));
        m_Name.resize(nameLen);
        stream.read(&m_Name[0], nameLen);
    }

	//////////////////////////////////////////////////////////////////////////
	float Tensor::TryGet(float def, int w, int h, int d, int n) const
	{
        if (h < 0 || h >= (int)Height() || w < 0 || w >= (int)Width() || d < 0 || d >= (int)Depth() || n < 0 || n >(int)Batch())
			return def;

		return Get(w, h, d, n);
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::TrySet(float value, int w, int h, int d, int n)
	{
		if (h < 0 || h >= (int)Height() || w < 0 || w >= (int)Width() || d < 0 || d >= (int)Depth() || n < 0 || n >(int)Batch())
			return;

		Set(value, w, h, d, n);
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::CopyTo(Tensor& result, float tau) const
	{
		assert(m_Shape.Length == result.m_Shape.Length);

        if (tau <= 0 && IsOnDevice() && result.IsOnDevice())
        {
            GetDeviceVar().CopyTo(result.GetDevicePtr());
            return;
        }

		CopyToHost();
        result.OverrideHost();

		if (tau <= 0)
			copy(m_Values.begin(), m_Values.end(), result.m_Values.begin());
		else
			Map([&](float v1, float v2) { return v1 * tau + v2 * (1 - tau); }, result, result);
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::CopyBatchTo(uint32_t batchId, uint32_t targetBatchId, Tensor& result) const
	{
        assert(SameDimensionsExceptBatches(result));
        assert(batchId < Batch());
        assert(targetBatchId < result.Batch());
		
        CopyToHost();
		result.OverrideHost();
        
		copy(m_Values.begin() + batchId * m_Shape.Dim0Dim1Dim2, 
			 m_Values.begin() + batchId * m_Shape.Dim0Dim1Dim2 + m_Shape.Dim0Dim1Dim2,
			 result.m_Values.begin() + targetBatchId * m_Shape.Dim0Dim1Dim2);
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::CopyDepthTo(uint32_t depthId, uint32_t batchId, uint32_t targetDepthId, uint32_t targetBatchId, Tensor& result) const
	{
		CopyToHost();
		result.OverrideHost();
		//if (m_Shape.Width != result.m_Shape.Width || m_Shape.Height != result.m_Shape.Height) throw new Exception("Incompatible tensors.");

		copy(m_Values.begin() + batchId * m_Shape.Dim0Dim1Dim2 + depthId * m_Shape.Dim0Dim1, 
			 m_Values.begin() + batchId * m_Shape.Dim0Dim1Dim2 + depthId * m_Shape.Dim0Dim1 + m_Shape.Dim0Dim1,
		     result.m_Values.begin() + targetBatchId * m_Shape.Dim0Dim1Dim2 + targetDepthId * m_Shape.Dim0Dim1);
	}

	//////////////////////////////////////////////////////////////////////////
	Tensor Tensor::GetBatch(uint32_t batchId) const
	{
		Tensor result(Shape(Width(), Height(), Depth()));
		CopyBatchTo(batchId, 0, result);
		return result;
	}

    //////////////////////////////////////////////////////////////////////////
    Tensor Tensor::GetBatches(vector<uint32_t> batchIds) const
    {
        Tensor result(Shape(Width(), Height(), Depth(), (uint32_t)batchIds.size()));
        GetBatches(batchIds, result);
        return result;
    }

    //////////////////////////////////////////////////////////////////////////
    void Tensor::GetBatches(vector<uint32_t> batchIds, Tensor& result) const
    {
        assert(SameDimensionsExceptBatches(result));
        assert(result.Batch() == (uint32_t)batchIds.size());

        for (size_t i = 0; i < batchIds.size(); ++i)
            CopyBatchTo(batchIds[i], (uint32_t)i, result);
    }

    //////////////////////////////////////////////////////////////////////////
    Tensor Tensor::GetRandomBatches(uint32_t batchSize) const
    {
        assert(batchSize <= Batch());

        vector<uint32_t> indices(Batch());
        iota(indices.begin(), indices.end(), 0);
        random_shuffle(indices.begin(), indices.end(), [&](size_t max) { return GlobalRng().Next((int)max); });
        indices.resize(batchSize);

        return GetBatches(indices);
    }

    //////////////////////////////////////////////////////////////////////////
	Tensor Tensor::GetDepth(uint32_t depthId, uint32_t batchId) const
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

		for (uint32_t i = 0; i < m_Values.size(); ++i)
			if (abs(m_Values[i] - other.m_Values[i]) > epsilon)
				return false;

		return true;
	}

    //////////////////////////////////////////////////////////////////////////
    template <int W, int H, int D, int N>
    Tensor MaxTemplate(const Tensor& input, Tensor* maxIndex)
    {
        auto& inputShape = input.GetShape();

        Tensor maxValue(Shape(W ? 1 : input.Width(), H ? 1 : input.Height(), D ? 1 : input.Depth(), N ? 1 : input.Batch()));
        maxValue.FillWithValue(-numeric_limits<float>().max());

        auto& minValueShape = maxValue.GetShape();

        auto& inputValues = input.GetValues();
        auto& minValueValues = maxValue.GetValues();

        if (maxIndex)
            maxIndex->FillWithValue(-1);

        size_t i = 0;
        for (uint32_t n = 0; n < input.Batch(); ++n)
        for (uint32_t d = 0; d < input.Depth(); ++d)
        for (uint32_t h = 0; h < input.Height(); ++h)
        for (uint32_t w = 0; w < input.Width(); ++w, ++i)
        {
            uint32_t idx = minValueShape.GetIndex(w * (1 - W), h * (1 - H), d * (1 - D), n * (1 - N));
            if (inputValues[i] > minValueValues[idx])
            {
                minValueValues[idx] = inputValues[i];
                if (maxIndex)
                {
                    if (W && H && D && N)
                        maxIndex->GetValues()[idx] = (float)inputShape.GetIndex(w, h, d, n);
                    else if (W && H && D)
                        maxIndex->GetValues()[idx] = (float)inputShape.GetIndex(w, h, d, 0u);
                    else
                        maxIndex->GetValues()[idx] = (float)(w * W + h * H + d * D + n * N);
                    // I didn't bother to come up with indices for other combinations :(
                }
            }
        }

        return maxValue;
    }

	//////////////////////////////////////////////////////////////////////////
    Tensor Tensor::Max(EAxis axis, Tensor* maxIndex) const
	{
        CopyToHost();

        if (axis == GlobalAxis)
            return MaxTemplate<1, 1, 1, 1>(*this, maxIndex);
        if (axis == WidthAxis)
            return MaxTemplate<1, 0, 0, 0>(*this, maxIndex);
        if (axis == HeightAxis)
            return MaxTemplate<0, 1, 0, 0>(*this, maxIndex);
        if (axis == DepthAxis)
            return MaxTemplate<0, 0, 1, 0>(*this, maxIndex);
        if (axis == BatchAxis)
            return MaxTemplate<0, 0, 0, 1>(*this, maxIndex);
        if (axis == _012Axes)
            return MaxTemplate<1, 1, 1, 0>(*this, maxIndex);
        if (axis == _013Axes)
            return MaxTemplate<1, 1, 0, 1>(*this, maxIndex);
        if (axis == _123Axes)
            return MaxTemplate<0, 1, 1, 1>(*this, maxIndex);

        assert(false);
        return Tensor();
	}

    //////////////////////////////////////////////////////////////////////////
    template <int W, int H, int D, int N>
    Tensor MinTemplate(const Tensor& input, Tensor* minIndex)
    {
        auto& inputShape = input.GetShape();

        Tensor minValue(Shape(W ? 1 : input.Width(), H ? 1 : input.Height(), D ? 1 : input.Depth(), N ? 1 : input.Batch()));
        minValue.FillWithValue(numeric_limits<float>().max());

        auto& minValueShape = minValue.GetShape();

        auto& inputValues = input.GetValues();
        auto& minValueValues = minValue.GetValues();

        if (minIndex)
            minIndex->FillWithValue(-1);

        size_t i = 0;
        for (uint32_t n = 0; n < input.Batch(); ++n)
        for (uint32_t d = 0; d < input.Depth(); ++d)
        for (uint32_t h = 0; h < input.Height(); ++h)
        for (uint32_t w = 0; w < input.Width(); ++w, ++i)
        {
            uint32_t idx = minValueShape.GetIndex(w * (1 - W), h * (1 - H), d * (1 - D), n * (1 - N));
            if (inputValues[i] < minValueValues[idx])
            {
                minValueValues[idx] = inputValues[i];
                if (minIndex)
                {
                    if (W && H && D && N)
                        minIndex->GetValues()[idx] = (float)inputShape.GetIndex(w, h, d, n);
                    else if (W && H && D)
                        minIndex->GetValues()[idx] = (float)inputShape.GetIndex(w, h, d, 0u);
                    else
                        minIndex->GetValues()[idx] = (float)(w * W + h * H + d * D + n * N);
                    // I didn't bother to come up with indices for other combinations :(
                }
            }
        }

        return minValue;
    }

    //////////////////////////////////////////////////////////////////////////
    Tensor Tensor::Min(EAxis axis, Tensor* minIndex) const
    {
        CopyToHost();

        if (axis == GlobalAxis)
            return MinTemplate<1, 1, 1, 1>(*this, minIndex);
        if (axis == WidthAxis)
            return MinTemplate<1, 0, 0, 0>(*this, minIndex);
        if (axis == HeightAxis)
            return MinTemplate<0, 1, 0, 0>(*this, minIndex);
        if (axis == DepthAxis)
            return MinTemplate<0, 0, 1, 0>(*this, minIndex);
        if (axis == BatchAxis)
            return MinTemplate<0, 0, 0, 1>(*this, minIndex);
        if (axis == _012Axes)
            return MinTemplate<1, 1, 1, 0>(*this, minIndex);
        if (axis == _013Axes)
            return MinTemplate<1, 1, 0, 1>(*this, minIndex);
        if (axis == _123Axes)
            return MinTemplate<0, 1, 1, 1>(*this, minIndex);
        
        assert(false);
        return Tensor();
    }

    //////////////////////////////////////////////////////////////////////////
    void Tensor::Sigmoid(Tensor& result) const
    {
        Op()->Sigmoid(*this, result);
    }

    //////////////////////////////////////////////////////////////////////////
    void Tensor::SigmoidGradient(const Tensor& output, const Tensor& outputGradient, Tensor& result) const
    {
        Op()->SigmoidGradient(output, outputGradient, result);
    }

    //////////////////////////////////////////////////////////////////////////
    void Tensor::Tanh(Tensor& result) const
    {
        Op()->Tanh(*this, result);
    }

    //////////////////////////////////////////////////////////////////////////
    void Tensor::TanhGradient(const Tensor& output, const Tensor& outputGradient, Tensor& result) const
    {
        Op()->TanhGradient(output, outputGradient, result);
    }

    //////////////////////////////////////////////////////////////////////////
    void Tensor::ReLU(Tensor& result) const
    {
        Op()->ReLU(*this, result);
    }

    //////////////////////////////////////////////////////////////////////////
    void Tensor::ReLUGradient(const Tensor& output, const Tensor& outputGradient, Tensor& result) const
    {
        Op()->ReLUGradient(output, outputGradient, result);
    }

	//////////////////////////////////////////////////////////////////////////
	void Tensor::Elu(float alpha, Tensor& result) const
	{
		Op()->Elu(*this, alpha, result);
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::EluGradient(const Tensor& output, const Tensor& outputGradient, float alpha, Tensor& result) const
	{
		Op()->EluGradient(output, outputGradient, alpha, result);
	}

    //////////////////////////////////////////////////////////////////////////
    void Tensor::LeakyReLU(float alpha, Tensor& result) const
    {
        Op()->LeakyReLU(*this, alpha, result);
    }

    //////////////////////////////////////////////////////////////////////////
    void Tensor::LeakyReLUGradient(const Tensor& output, const Tensor& outputGradient, float alpha, Tensor& result) const
    {
        Op()->LeakyReLUGradient(output, outputGradient, alpha, result);
    }

	//////////////////////////////////////////////////////////////////////////
	void Tensor::Softmax(Tensor& result) const
	{
		Op()->Softmax(*this, result);
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::SoftmaxGradient(const Tensor& output, const Tensor& outputGradient, Tensor& result) const
	{
		Op()->SoftmaxGradient(output, outputGradient, result);
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
    TensorOpCpu* Tensor::DefaultOp()
    {
        if (g_DefaultOp == nullptr)
            g_DefaultOp = g_OpCpu;

        return g_DefaultOp;
    }

    //////////////////////////////////////////////////////////////////////////
    TensorOpCpu* Tensor::ActiveOp()
    {
        return g_ForcedOp ? g_ForcedOp : DefaultOp();
    }

    //////////////////////////////////////////////////////////////////////////
    void Tensor::DebugDumpValues(const string& outFile) const
    {
        CopyToHost();
        ofstream stream(outFile);
        for (int i = 0; i < 4; ++i)
            stream << m_Shape.Dimensions[i] << "\n";
        for (int i = 0; i < m_Values.size(); ++i)
            stream << m_Values[i] << "\n";
        stream.close();
    }

    //////////////////////////////////////////////////////////////////////////
    void Tensor::DebugRecoverValues(const string& inFile)
    {
        OverrideHost();
        ifstream stream(inFile);
        vector<int> dimensions(4);
        for (int i = 0; i < 4; ++i)
            stream >> dimensions[i];
        m_Shape = Shape::From(dimensions);
        m_Values.resize(m_Shape.Length);
        for (int i = 0; i < m_Values.size(); ++i)
            stream >> m_Values[i];
        stream.close();
    }

    //////////////////////////////////////////////////////////////////////////
	Neuro::TensorOpCpu* Tensor::GetOpFromMode(EOpMode mode)
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
        Release();
    }

    //////////////////////////////////////////////////////////////////////////
    void Tensor::GPUData::Release()
    {
        delete(m_DeviceVar);
        m_DeviceVar = nullptr;
        delete(m_ConvWorkspace);
        m_ConvWorkspace = nullptr;
        delete(m_ConvBackWorkspace);
        m_ConvBackWorkspace = nullptr;
        delete(m_ConvBackKernelWorkspace);
        m_ConvBackKernelWorkspace = nullptr;
        delete(m_DropoutWorkspace);
        m_DropoutWorkspace = nullptr;
        delete(m_DropoutStates);
        m_DropoutStates = nullptr;
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

    //////////////////////////////////////////////////////////////////////////
    Tensor operator*(const Tensor& t1, const Tensor& t2)
    {
        return t1.MulElem(t2);
    }

    //////////////////////////////////////////////////////////////////////////
    Tensor operator*(const Tensor& t, float v)
    {
        return t.Mul(v);
    }

    //////////////////////////////////////////////////////////////////////////
    Tensor operator/(const Tensor& t, float v)
    {
        return t.Div(v);
    }

    //////////////////////////////////////////////////////////////////////////
    Tensor operator/(float v, const Tensor& t)
    {
        return t.Map([&](float x) { return v / x; });
    }

    //////////////////////////////////////////////////////////////////////////
    Tensor operator+(const Tensor& t1, const Tensor& t2)
    {
        return t1.Add(t2);
    }

    //////////////////////////////////////////////////////////////////////////
    Tensor operator+(const Tensor& t1, float v)
    {
        return t1.Add(v);
    }

    //////////////////////////////////////////////////////////////////////////
    Tensor operator-(const Tensor& t1, const Tensor& t2)
    {
        return t1.Sub(t2);
    }

    //////////////////////////////////////////////////////////////////////////
    Tensor operator-(const Tensor& t, float v)
    {
        return t.Sub(v);
    }

    //////////////////////////////////////////////////////////////////////////
    Tensor operator-(const Tensor& t)
    {
        return t.Negated();
    }

    //////////////////////////////////////////////////////////////////////////
    Tensor pow(const Tensor& t, float p)
    {
        return t.Map([&](float x) { return ::pow(x, p); });
    }

    //////////////////////////////////////////////////////////////////////////
    Tensor sqr(const Tensor& t)
    {
        return t.Map([&](float x) { return x * x; });
    }

    //////////////////////////////////////////////////////////////////////////
    Tensor sqrt(const Tensor& t)
    {
        return t.Map([](float x) { return ::sqrt(x); });
    }

    //////////////////////////////////////////////////////////////////////////
    Tensor sum(const Tensor& t, EAxis axis)
    {
        return t.Sum(axis);
    }

    //////////////////////////////////////////////////////////////////////////
    Tensor mean(const Tensor& t, EAxis axis)
    {
        return t.Mean(axis);
    }

    //////////////////////////////////////////////////////////////////////////
    Tensor zeros(const Shape& shape)
    {
        return Tensor(shape).FillWithValue(0);
    }

    //////////////////////////////////////////////////////////////////////////
    Tensor ones(const Shape& shape)
    {
        return Tensor(shape).FillWithValue(1);
    }
}
