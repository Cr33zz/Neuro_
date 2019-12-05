#include <algorithm>
#include <fstream>
#include <numeric>
#include <experimental/filesystem>
#include <FreeImage.h>
#include <H5Cpp.h>

#include "Tensors/Tensor.h"
#include "Tensors/TensorOpCpu.h"
#include "Tensors/TensorOpMultiCpu.h"
#include "Tensors/TensorOpGpu.h"
#include "Tensors/TensorFormatter.h"
#include "Random.h"
#include "Tools.h"

namespace Neuro
{
    using namespace H5;
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
    Tensor::Tensor(const Shape& shape, const string& name, EStorageType storageType)
        : m_Name(name), m_Shape(shape), m_Storage(storageType, shape.Length, name)
    {
        m_Shape = shape;
        m_Op = DefaultOp();
    }

    //////////////////////////////////////////////////////////////////////////
    Tensor::Tensor(istream& stream, EStorageType storageType)
        : Tensor(Shape(0), "", storageType)
    {
        LoadBin(stream);
    }

	//////////////////////////////////////////////////////////////////////////
	Tensor::Tensor(const Tensor& t)
	{
        *this = t;
	}

    //////////////////////////////////////////////////////////////////////////
    Tensor::Tensor(Tensor&& t)
    {
        *this = move(t);
    }

    //////////////////////////////////////////////////////////////////////////
    Tensor& Tensor::operator=(const Tensor& t)
    {
        if (this != &t)
        {
            m_Storage = t.m_Storage;
            m_Shape = t.m_Shape;
            m_Op = t.m_Op;
        }
        return *this;
    }

    //////////////////////////////////////////////////////////////////////////
    Tensor& Tensor::operator=(Tensor&& t)
    {
        if (this != &t)
        {
            m_Storage = move(t.m_Storage);
            m_Name = t.m_Name;
            m_Shape = t.m_Shape;
            m_Op = t.m_Op;
        }
        return *this;
    }

	//////////////////////////////////////////////////////////////////////////
    Tensor::Tensor(const vector<float>& values, const string& name, EStorageType storageType)
		: Tensor(values, Shape((int)values.size()), name, storageType)
	{
	}

	//////////////////////////////////////////////////////////////////////////
    Tensor::Tensor(const vector<float>& values, const Shape& shape, const string& name, EStorageType storageType)
		: Tensor(shape, name, storageType)
	{
		assert(values.size() == shape.Length);// && string("Invalid array size ") + to_string(values.size()) + ". Expected " + to_string(shape.Length) + ".");
        memcpy(m_Storage.Data(), &values[0], values.size() * sizeof(float));
	}

	//////////////////////////////////////////////////////////////////////////
    Tensor::Tensor(const string& imageFile, bool normalize, bool grayScale, const string& name, EStorageType storageType)
        : Tensor(Shape(0), name, storageType)
	{
        ImageLibInit();

        auto format = GetFormat(imageFile);
        assert(format != FIF_UNKNOWN);

        FIBITMAP* image = FreeImage_Load(format, imageFile.c_str());

        assert(image);

        const uint32_t WIDTH = FreeImage_GetWidth(image);
        const uint32_t HEIGHT = FreeImage_GetHeight(image);

        m_Shape = Shape(WIDTH, HEIGHT, grayScale ? 1 : 3);
        m_Storage.Resize(m_Shape.Length);

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
    void Tensor::SaveAsImage(const string& imageFile, bool denormalize, uint32_t maxCols) const
    {
        ImageLibInit();

        auto format = GetFormat(imageFile);
        assert(format != FIF_UNKNOWN);

        const uint32_t TENSOR_WIDTH = Width();
        const uint32_t TENSOR_HEIGHT = Height();
        const uint32_t IMG_COLS = min((uint32_t)ceil(::sqrt((float)Batch())), maxCols == 0 ? numeric_limits<uint32_t>().max() : maxCols);
        const uint32_t IMG_ROWS = (uint32_t)ceil((float)Batch() / IMG_COLS);
        const uint32_t IMG_WIDTH = IMG_COLS * TENSOR_WIDTH;
        const uint32_t IMG_HEIGHT = IMG_ROWS * TENSOR_HEIGHT;
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
    void Tensor::SaveAsH5(const string& h5File) const
    {
        SyncToHost();
        H5File file = H5File(h5File, H5F_ACC_TRUNC);
        Group g(file.createGroup(Name().empty() ? "g" : Name()));

        const auto& shape = GetShape();
        vector<hsize_t> dims;
        for (uint32_t i = 0; i < shape.NDim; ++i)
            dims.push_back(shape.Dimensions[i]);

        DataSet dataset(g.createDataSet("data", PredType::NATIVE_FLOAT, DataSpace(shape.NDim, &dims[0])));
        dataset.write(Values(), PredType::NATIVE_FLOAT);
    }

    //////////////////////////////////////////////////////////////////////////
    void Tensor::LoadFromH5(const string& h5File)
    {
        if (!std::experimental::filesystem::exists(h5File))
        {
            cout << "File '" << h5File << "' does not exist.\n";
            return;
        }

        if (!H5File::isHdf5(h5File.c_str()))
        {
            cout << "File '" << h5File << "' is not valid HDF5 file.\n";
            return;
        }

        OverrideHost();
        H5File file = H5File(h5File, H5F_ACC_RDONLY);

        hsize_t groupsNum;
        H5Gget_num_objs(file.getId(), &groupsNum);

        NEURO_ASSERT(groupsNum == 1, "Expected exactly 1 group, found " << groupsNum << ".");

        Group g(file.openGroup(file.getObjnameByIdx(0)));

        auto dataset = g.openDataSet("data");

        hsize_t nDims = dataset.getSpace().getSimpleExtentNdims();
        hsize_t dims[5];
        dataset.getSpace().getSimpleExtentDims(nullptr, dims);

        Resize(Shape::From({ (int)dims[0], nDims > 1 ? (int)dims[1] : 1, nDims > 2 ? (int)dims[2] : 1, nDims > 3 ? (int)dims[3] : 1 }));

        dataset.read(Values(), PredType::NATIVE_FLOAT);
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
    void Tensor::Name(const string& name)
    {
        m_Name = name;
        m_Storage.Rename(name);
    }

    //////////////////////////////////////////////////////////////////////////
	float* Tensor::Values()
	{
		CopyToHost(true);
		return m_Storage.Data();
	}

    //////////////////////////////////////////////////////////////////////////
    const float* Tensor::Values() const
    {
        CopyToHost();
        return m_Storage.Data();
    }

    //////////////////////////////////////////////////////////////////////////
    void Tensor::SetStorageType(int type)
    {
        m_Storage.ChangeType(type);
    }

    //////////////////////////////////////////////////////////////////////////
    Tensor& Tensor::FillWithRand(int seed, float min, float max, uint32_t offset)
	{
		OverrideHost();

		auto fillUp = [&](Random& rng)
		{
			for (uint32_t i = offset; i < m_Storage.Size(); ++i)
				m_Storage.Data()[i] = min + (max - min) * rng.NextFloat();
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
        for (uint32_t i = offset; i < m_Storage.Size(); ++i)
			m_Storage.Data()[i] = start + i * increment;
		return *this;
	}

	//////////////////////////////////////////////////////////////////////////
	Tensor& Tensor::FillWithValue(float value, uint32_t offset)
	{
		OverrideHost();
		for (uint32_t i = offset; i < m_Storage.Size(); ++i)
			m_Storage.Data()[i] = value;
		return *this;
	}

    //////////////////////////////////////////////////////////////////////////
    Tensor& Tensor::FillWithFunc(const function<float()>& func, uint32_t offset)
    {
        OverrideHost();
        for (uint32_t i = offset; i < m_Storage.Size(); ++i)
            m_Storage.Data()[i] = func();
        return *this;
    }

    //////////////////////////////////////////////////////////////////////////
	void Tensor::Zero()
	{
        if (m_Storage.Location() == None)
            OverrideHost();
        
        if (m_Storage.Location() == Host)
            g_OpCpu->Zero(*this);
        else
            g_OpGpu->Zero(*this);
	}

    //////////////////////////////////////////////////////////////////////////
    void Tensor::One()
    {
        if (m_Storage.Location() == None)
            OverrideHost();

        if (m_Storage.Location() == Host)
            g_OpCpu->One(*this);
        else
            g_OpGpu->One(*this);
    }

    //////////////////////////////////////////////////////////////////////////
    Tensor Tensor::ToNCHW() const
    {
        CopyToHost();
        Tensor result(GetShape());

        uint32_t i = 0;
        for (uint32_t n = 0; n < Batch(); ++n)
        for (uint32_t h = 0; h < Height(); ++h)
        for (uint32_t w = 0; w < Width(); ++w)
        {
            for (uint32_t j = 0; j < Depth(); ++j, ++i)
            {
                result(w, h, j, n) = m_Storage.Data()[i];
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
                result.m_Storage.Data()[i] = Get(w, h, j, n);
            }
        }

        return result;
    }

    //////////////////////////////////////////////////////////////////////////
    Tensor Tensor::ToGrayScale() const
    {
        NEURO_ASSERT(Depth() == 3, "Expected 3 color channels.");
        NEURO_ASSERT(Batch() == 1, "Batches are not supported.");
        SyncToHost();
        Tensor output(Shape(Width(), Height()));

        uint32_t i = 0;
        for (uint32_t h = 0; h < Height(); ++h)
        for (uint32_t w = 0; w < Width(); ++w, ++i)
            output.Values()[i] = Get(w, h, 0) * 0.2989f + Get(w, h, 1) * 0.5870f + Get(w, h, 2) * 0.1140f;

        return output;
    }

    //////////////////////////////////////////////////////////////////////////
    Tensor Tensor::ToRGB() const
    {
        NEURO_ASSERT(Depth() == 1, "Expected 1 color channels.");
        NEURO_ASSERT(Batch() == 1, "Batches are not supported.");
        SyncToHost();
        Tensor output(Shape(Width(), Height(), 3));

        uint32_t i = 0;
        for (uint32_t h = 0; h < Height(); ++h)
        for (uint32_t w = 0; w < Width(); ++w, ++i)
        {
            output(w, h, 0) = Values()[i];
            output(w, h, 1) = Values()[i];
            output(w, h, 2) = Values()[i];
        }

        return output;
    }

    //////////////////////////////////////////////////////////////////////////
	void Tensor::MatMul(bool transposeT, const Tensor& t, Tensor& output) const
	{
        NEURO_ASSERT(!transposeT, "Fused tranpose and matmul not supported.");
        NEURO_ASSERT((!transposeT && Width() == t.Height()) || (transposeT && Width() == t.Width()), "");
		NEURO_ASSERT(t.Depth() == Depth(), "Depths must match.");
        NEURO_ASSERT(output.Batch() == max(Batch(), t.Batch()), "Output batch size doesn't match maximum of input tensors' batch sizes.");

		Op()->MatMul(false, transposeT, *this, t, output);
	}

	//////////////////////////////////////////////////////////////////////////
	Tensor Tensor::MatMul(bool transposeT, const Tensor& t) const
	{
		Tensor result(Shape(transposeT ? t.m_Shape.Height() : t.m_Shape.Width(), Height(), Depth(), max(Batch(), t.Batch())));
		MatMul(transposeT, t, result);
		return result;
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::MatMul(const Tensor& t, Tensor& result) const
	{
		MatMul(false, t, result);
	}

	//////////////////////////////////////////////////////////////////////////
	Tensor Tensor::MatMul(const Tensor& t) const
	{
		return MatMul(false, t);
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::MulElem(const Tensor& t, Tensor& result) const
	{
		Op()->Mul(1.f, *this, 1.f, t, result);
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
        Op()->Mul(*this, v, result);
	}

	//////////////////////////////////////////////////////////////////////////
	Tensor Tensor::Mul(float v) const
	{
		Tensor result(m_Shape);
		Mul(v, result);
		return result;
	}

    //////////////////////////////////////////////////////////////////////////
    void Tensor::Scale(float v)
    {
        Op()->Scale(*this, v);
    }

    //////////////////////////////////////////////////////////////////////////
	void Tensor::Div(const Tensor& t, Tensor& result) const
	{
        Op()->Div(1.f, *this, 1.f, t, result);
	}

    //////////////////////////////////////////////////////////////////////////
    void Tensor::Div(float alpha, float beta, const Tensor& t, Tensor& result) const
    {
        Op()->Div(alpha, *this, beta, t, result);
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
        Op()->Add(*this, v, result);
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
        Add(-v, result);
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
        Op()->Negate(*this, result);
	}

	//////////////////////////////////////////////////////////////////////////
	Tensor Tensor::Negated() const
	{
		Tensor result(m_Shape);
		Negated(result);
		return result;
	}

    //////////////////////////////////////////////////////////////////////////
    void Tensor::Inversed(float alpha, Tensor& result) const
    {
        Op()->Inverse(alpha, *this, result);
    }

    //////////////////////////////////////////////////////////////////////////
    Tensor Tensor::Inversed(float alpha) const
    {
        Tensor result(m_Shape);
        Inversed(alpha, result);
        return result;
    }

	//////////////////////////////////////////////////////////////////////////
	void Tensor::Clip(float min, float max, Tensor& result) const
	{
        Op()->Clip(*this, min, max, result);
	}

	//////////////////////////////////////////////////////////////////////////
	Tensor Tensor::Clip(float min, float max) const
	{
		Tensor result(m_Shape);
		Clip(min, max, result);
		return result;
	}

    //////////////////////////////////////////////////////////////////////////
    void Tensor::ClipGradient(const Tensor& input, float min, float max, const Tensor& outputGradient, Tensor& inputGradient) const
    {
        Op()->ClipGradient(input, min, max, outputGradient, inputGradient);
    }

    //////////////////////////////////////////////////////////////////////////
    void Tensor::ExtractSubTensor2D(uint32_t widthOffset, uint32_t heightOffset, Tensor& output) const
    {
        Op()->ExtractSubTensor2D(*this, widthOffset, heightOffset, output);
    }

    //////////////////////////////////////////////////////////////////////////
    void Tensor::FuseSubTensor2D(uint32_t widthOffset, uint32_t heightOffset, Tensor& output) const
    {
        Op()->FuseSubTensor2D(*this, widthOffset, heightOffset, output);
    }

    //////////////////////////////////////////////////////////////////////////
	Tensor Tensor::DiagFlat() const
	{
        CopyToHost();
		Tensor result(zeros(Shape(BatchLength(), BatchLength(), 1, Batch())));

        uint32_t batchLen = BatchLength();

		for (uint32_t b = 0; b < Batch(); ++b)
		for (uint32_t i = 0; i < batchLen; ++i)
			result(i, i, 0, b) = m_Storage.Data()[b * batchLen + i];

		return result;
	}

    //////////////////////////////////////////////////////////////////////////
    Tensor Tensor::Pow(float power) const
    {
        Tensor result(m_Shape);
        Pow(power, result);
        return result;
    }

    //////////////////////////////////////////////////////////////////////////
    void Tensor::Pow(float power, Tensor& result) const
    {
        Op()->Pow(*this, power, result);
    }

    //////////////////////////////////////////////////////////////////////////
    void Tensor::PowGradient(const Tensor& input, float power, const Tensor& outputGradient, Tensor& inputGradient) const
    {
        Op()->PowGradient(input, power, outputGradient, inputGradient);
    }

    //////////////////////////////////////////////////////////////////////////
    Tensor Tensor::Abs() const
    {
        Tensor result(m_Shape);
        Abs(result);
        return result;
    }

    //////////////////////////////////////////////////////////////////////////
    void Tensor::Abs(Tensor& result) const
    {
        Op()->Abs(*this, result);
    }

    //////////////////////////////////////////////////////////////////////////
    void Tensor::AbsGradient(const Tensor& input, const Tensor& outputGradient, Tensor& inputGradient) const
    {
        Op()->AbsGradient(input, outputGradient, inputGradient);
    }

    //////////////////////////////////////////////////////////////////////////
    Tensor Tensor::Sqrt() const
    {
        Tensor result(m_Shape);
        Sqrt(result);
        return result;
    }

    //////////////////////////////////////////////////////////////////////////
    void Tensor::Sqrt(Tensor& output) const
    {
        Op()->Sqrt(*this, output);
    }

    //////////////////////////////////////////////////////////////////////////
    Tensor Tensor::Log() const
    {
        Tensor result(m_Shape);
        Log(result);
        return result;
    }

    //////////////////////////////////////////////////////////////////////////
    void Tensor::Log(Tensor& output) const
    {
        Op()->Log(*this, output);
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
    template <int W, int H, int D, int N, bool ABS>
    Tensor SumTemplate(const Tensor& input, EAxis axis)
    {
        Tensor sum(Shape(W ? 1 : input.Width(), H ? 1 : input.Height(), D ? 1 : input.Depth(), N ? 1 : input.Batch()));
        if (ABS)
            input.AbsSum(axis, sum);
        else
            input.Sum(axis, sum);
        return sum;
    }

    //////////////////////////////////////////////////////////////////////////
    Tensor Tensor::AbsSum(EAxis axis) const
    {
        if (axis == GlobalAxis)
            return SumTemplate<1, 1, 1, 1, true>(*this, axis);
        if (axis == WidthAxis)
            return SumTemplate<1, 0, 0, 0, true>(*this, axis);
        if (axis == HeightAxis)
            return SumTemplate<0, 1, 0, 0, true>(*this, axis);
        if (axis == DepthAxis)
            return SumTemplate<0, 0, 1, 0, true>(*this, axis);
        if (axis == BatchAxis)
            return SumTemplate<0, 0, 0, 1, true>(*this, axis);
        if (axis == _01Axes)
            return SumTemplate<1, 1, 0, 0, true>(*this, axis);
        if (axis == _012Axes)
            return SumTemplate<1, 1, 1, 0, true>(*this, axis);
        if (axis == _013Axes)
            return SumTemplate<1, 1, 0, 1, true>(*this, axis);
        if (axis == _123Axes)
            return SumTemplate<0, 1, 1, 1, true>(*this, axis);

        assert(false);
        return Tensor();
    }

    //////////////////////////////////////////////////////////////////////////
    void Tensor::AbsSum(EAxis axis, Tensor& output) const
    {
        Op()->AbsSum(*this, axis, output);
    }

	//////////////////////////////////////////////////////////////////////////
    Tensor Tensor::Sum(EAxis axis) const
	{
        if (axis == GlobalAxis)
            return SumTemplate<1, 1, 1, 1, false>(*this, axis);
        if (axis == WidthAxis)
            return SumTemplate<1, 0, 0, 0, false>(*this, axis);
        if (axis == HeightAxis)
            return SumTemplate<0, 1, 0, 0, false>(*this, axis);
        if (axis == DepthAxis)
            return SumTemplate<0, 0, 1, 0, false>(*this, axis);
        if (axis == BatchAxis)
            return SumTemplate<0, 0, 0, 1, false>(*this, axis);
        if (axis == _01Axes)
            return SumTemplate<1, 1, 0, 0, false>(*this, axis);
        if (axis == _012Axes)
            return SumTemplate<1, 1, 1, 0, false>(*this, axis);
        if (axis == _013Axes)
            return SumTemplate<1, 1, 0, 1, false>(*this, axis);
        if (axis == _123Axes)
            return SumTemplate<0, 1, 1, 1, false>(*this, axis);

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
    Shape MeanShape(const Tensor& input, EAxis axis)
    {
        return Shape(W ? 1 : input.Width(), H ? 1 : input.Height(), D ? 1 : input.Depth(), N ? 1 : input.Batch());
    }

    //////////////////////////////////////////////////////////////////////////
    Tensor Tensor::Mean(EAxis axis) const
	{
        Shape outputShape;
        switch (axis)
        {
        case Neuro::GlobalAxis:
            outputShape = MeanShape<1, 1, 1, 1>(*this, axis);
            break;
        case Neuro::WidthAxis:
            outputShape = MeanShape<1, 0, 0, 0>(*this, axis);
            break;
        case Neuro::HeightAxis:
            outputShape = MeanShape<0, 1, 0, 0>(*this, axis);
            break;
        case Neuro::DepthAxis:
            outputShape = MeanShape<0, 0, 1, 0>(*this, axis);
            break;
        case Neuro::BatchAxis:
            outputShape = MeanShape<0, 0, 0, 1>(*this, axis);
            break;
        case Neuro::_01Axes:
            outputShape = MeanShape<1, 1, 0, 0>(*this, axis);
            break;
        case Neuro::_012Axes:
            outputShape = MeanShape<1, 1, 1, 0>(*this, axis);
            break;
        case Neuro::_013Axes:
            outputShape = MeanShape<1, 1, 0, 1>(*this, axis);
            break;
        case Neuro::_123Axes:
            outputShape = MeanShape<0, 1, 1, 1>(*this, axis);
            break;
        default:
            NEURO_ASSERT(false, "Unsupported axis.");
            break;
        }
        Tensor output(outputShape);
        Mean(axis, output);
        return output;
	}

    //////////////////////////////////////////////////////////////////////////
    void Tensor::Mean(EAxis axis, Tensor& output) const
    {
        Op()->Mean(*this, axis, output);
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
			copy(t.m_Storage.Data(), t.m_Storage.DataEnd(), output.m_Storage.Data() + t.Length() * n);
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
			copy(t.m_Storage.Data(), t.m_Storage.DataEnd(), output.m_Storage.Data() + t.Length() * n);
		}

		for (uint32_t n = t0_copies; n < output.Depth(); ++n)
		{
			const Tensor& t = tensors[n - t0_copies];
			t.CopyToHost();
			copy(t.m_Storage.Data(), t.m_Storage.DataEnd(), output.m_Storage.Data() + t.Length() * n);
		}

		return output;
	}

    //////////////////////////////////////////////////////////////////////////
    template <int W, int H, int D, int N>
    void ConcatTemplate(const const_tensor_ptr_vec_t& inputs, Tensor& output)
    {
        output.OverrideHost();
        auto& shape = inputs[0]->GetShape();
        const uint32_t width = shape.Width();
        const uint32_t height = shape.Height();
        const uint32_t depth = shape.Depth();
        const uint32_t batch = shape.Batch();

        for (uint32_t i = 0; i < (uint32_t)inputs.size(); ++i)
        {
            inputs[i]->CopyToHost();
            auto inputValues = inputs[i]->Values();
            size_t j = 0;
            for (uint32_t n = 0; n < batch; ++n)
            for (uint32_t d = 0; d < depth; ++d)
            for (uint32_t h = 0; h < height; ++h)
            for (uint32_t w = 0; w < width; ++w, ++j)
                output(w + (W ? width * i : 0), h + (H ? height * i : 0), d + (D ? depth * i : 0), n + (N ? batch * i : 0)) = inputValues[j];
        }
    }

	//////////////////////////////////////////////////////////////////////////
	void Tensor::Concat(EAxis axis, const const_tensor_ptr_vec_t& inputs, Tensor& output)
	{
        if (axis == BatchAxis)
        {
            NEURO_ASSERT(output.Batch() == (uint32_t)inputs.size(), "Invalid output batch dimension, expected " << inputs.size());
            uint32_t elementsCopied = 0;
            for (uint32_t i = 0; i < inputs.size(); ++i)
            {
                NEURO_ASSERT(inputs[i]->Batch() == 1, "");
                inputs[i]->CopyTo(0, output, elementsCopied, inputs[i]->Length());
                elementsCopied += inputs[i]->Length();
            }
            return;
        }

        if (axis == DepthAxis)
        {
            for (uint32_t n = 0; n < output.Batch(); ++n)
            {
                uint32_t elementsCopied = 0;
                for (uint32_t i = 0; i < inputs.size(); ++i)
                {
                    NEURO_ASSERT(inputs[i]->Batch() == output.Batch(), "");
                    inputs[i]->CopyTo(n * inputs[i]->BatchLength(), output, n * output.BatchLength() + elementsCopied, inputs[i]->BatchLength());
                    elementsCopied += inputs[i]->Stride(3);
                }
                NEURO_ASSERT(output.BatchLength() == elementsCopied, "Invalid output depth dimension, expected " << elementsCopied);
            }
            return;
        }

        if (axis == WidthAxis)
        {
            for (uint32_t n = 0; n < output.Batch(); ++n)
            for (uint32_t d = 0; d < output.Depth(); ++d)
            for (uint32_t h = 0; h < output.Height(); ++h)
            {
                uint32_t elementsCopied = 0;
                for (uint32_t i = 0; i < inputs.size(); ++i)
                {
                    NEURO_ASSERT(inputs[i]->Batch() == output.Batch(), "");
                    uint32_t inOffset = n * inputs[i]->Stride(3) + d * inputs[i]->Stride(2) + h * inputs[i]->Stride(1);
                    uint32_t outOffset = n * output.Stride(3) + d * output.Stride(2) + h * output.Stride(1);
                    inputs[i]->CopyTo(inOffset, output, outOffset + elementsCopied, inputs[i]->Width());
                    elementsCopied += inputs[i]->Width();
                }
                NEURO_ASSERT(output.Width() == elementsCopied, "Invalid output width dimension, expected " << elementsCopied);
            }
            return;
        }

        if (axis == HeightAxis)
            return ConcatTemplate<0, 1, 0, 0>(inputs, output);
        
        NEURO_ASSERT(false, "Unsupported axis.");
	}

    //////////////////////////////////////////////////////////////////////////
    template <int W, int H, int D, int N>
    void SplitTemplate(const Tensor& input, tensor_ptr_vec_t& outputs)
    {
        input.SyncToHost();
        auto& shape = outputs[0]->GetShape();
        const uint32_t width = shape.Width();
        const uint32_t height = shape.Height();
        const uint32_t depth = shape.Depth();
        const uint32_t batch = shape.Batch();

        for (uint32_t i = 0; i < (uint32_t)outputs.size(); ++i)
        {
            outputs[i]->OverrideHost();
            auto outputValues = outputs[i]->Values();
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
        if (axis == BatchAxis)
        {
            NEURO_ASSERT(Batch() == (uint32_t)outputs.size(), "Invalid batch dimension, expected " << outputs.size());
            uint32_t elementsCopied = 0;
            uint32_t singleOutputLen = Length() / (uint32_t)outputs.size();

            for (uint32_t i = 0; i < outputs.size(); ++i)
            {
                NEURO_ASSERT(outputs[i]->Batch() == 1, "");
                CopyTo(elementsCopied, *outputs[i], 0, singleOutputLen);
                elementsCopied += singleOutputLen;
            }
            return;
        }

        if (axis == DepthAxis)
        {
            for (uint32_t n = 0; n < Batch(); ++n)
            {
                uint32_t elementsCopied = 0;
                for (uint32_t i = 0; i < outputs.size(); ++i)
                {
                    NEURO_ASSERT(outputs[i]->Batch() == Batch(), "");
                    CopyTo(n * BatchLength() + elementsCopied, *outputs[i], n * outputs[i]->BatchLength(), outputs[i]->BatchLength());
                    elementsCopied += outputs[i]->BatchLength();
                }
                NEURO_ASSERT(BatchLength() == elementsCopied, "Invalid depth dimension, expected " << elementsCopied);
            }
            return;
        }

        if (axis == WidthAxis)
        {
            for (uint32_t n = 0; n < Batch(); ++n)
            for (uint32_t d = 0; d < Depth(); ++d)
            for (uint32_t h = 0; h < Height(); ++h)
            {
                uint32_t elementsCopied = 0;
                for (uint32_t i = 0; i < outputs.size(); ++i)
                {
                    NEURO_ASSERT(outputs[i]->Batch() == Batch(), "");
                    uint32_t outOffset = n * outputs[i]->Stride(3) + d * outputs[i]->Stride(2) + h * outputs[i]->Stride(1);
                    uint32_t offset = n * Stride(3) + d * Stride(2) + h * Stride(1);
                    CopyTo(offset + elementsCopied, *outputs[i], outOffset, outputs[i]->Width());
                    elementsCopied += outputs[i]->Width();
                }
                NEURO_ASSERT(Width() == elementsCopied, "Invalid output width dimension, expected " << elementsCopied);
            }
            return;
        }
        
        if (axis == HeightAxis)
            return SplitTemplate<0, 1, 0, 0>(*this, outputs);
        
        NEURO_ASSERT(false, "Unsupported axis.");
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::MergeMin(const const_tensor_ptr_vec_t& inputs, Tensor& output)
	{
        output.OverrideHost();
		inputs[0]->CopyTo(output);
		for (uint32_t i = 1; i < inputs.size(); ++i)
		for (uint32_t j = 0; j < output.Length(); ++j)
			output.m_Storage.Data()[j] = output.m_Storage.Data()[j] > inputs[i]->m_Storage.Data()[j] ? inputs[i]->m_Storage.Data()[j] : output.m_Storage.Data()[j];
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::MergeMax(const const_tensor_ptr_vec_t& inputs, Tensor& output)
	{
        output.OverrideHost();
        inputs[0]->CopyToHost();
		inputs[0]->CopyTo(output);

        for (uint32_t i = 1; i < inputs.size(); ++i)
        {
            inputs[i]->CopyToHost();
            for (uint32_t j = 0; j < output.Length(); ++j)
                output.m_Storage.Data()[j] = output.m_Storage.Data()[j] < inputs[i]->m_Storage.Data()[j] ? inputs[i]->m_Storage.Data()[j] : output.m_Storage.Data()[j];
        }
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::MergeSum(const const_tensor_ptr_vec_t& inputs, Tensor& output)
	{
        //output.OverrideHost();
		output.Zero();
        for (uint32_t i = 0; i < inputs.size(); ++i)
        {
            inputs[i]->Add(1.f, 1.f, output, output);
            /*inputs[i]->CopyToHost();
            for (uint32_t j = 0; j < result.Length(); ++j)
                result.m_Storage.Data()[j] += inputs[i]->m_Storage.Data()[j];*/
        }
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
            inputs[i]->CopyToHost();
            results[i]->OverrideHost();
			results[i]->Zero();
			for (uint32_t j = 0; j < output.Length(); ++j)
				results[i]->m_Storage.Data()[j] = inputs[i]->m_Storage.Data()[j] == output.m_Storage.Data()[j] ? outputGradient.m_Storage.Data()[j] : 0;
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
                    norm(n) += normMode == ENormMode::L1 ? abs(m_Storage.Data()[idx]) : (m_Storage.Data()[idx] * m_Storage.Data()[idx]);

                if (normMode == ENormMode::L2)
                {
                    for (uint32_t i = 0; i < norm.Length(); ++i)
                        norm.m_Storage.Data()[i] = sqrt(norm.m_Storage.Data()[i]);
                }
            }
        
            for (uint32_t n = 0; n < Batch(); ++n)
            for (uint32_t i = 0, idx = n * BatchLength(); i < BatchLength(); ++i, ++idx)
                result.m_Storage.Data()[idx] = m_Storage.Data()[idx] / norm(n);

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
                        norm.m_Storage.Data()[i] = ::sqrt(norm.m_Storage.Data()[i]);
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
                    norm(0) += normMode == ENormMode::L1 ? abs(m_Storage.Data()[i]) : (m_Storage.Data()[i] * m_Storage.Data()[i]);

                if (normMode == ENormMode::L2)
                {
                    for (uint32_t i = 0; i < norm.Length(); ++i)
                        norm.m_Storage.Data()[i] = ::sqrt(norm.m_Storage.Data()[i]);
                }
            }

            for (uint32_t i = 0; i < Length(); ++i)
                result.m_Storage.Data()[i] = m_Storage.Data()[i] / norm(0);

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
                    result.m_Storage.Data()[idx] = rangeSpread * (m_Storage.Data()[idx] - minVal) / spread + scaleMin;
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
                result.m_Storage.Data()[i] = rangeSpread * (m_Storage.Data()[i] - minVal) / spread + scaleMin;
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
    void Tensor::ConstantPad2D(uint32_t left, uint32_t right, uint32_t top, uint32_t bottom, float value, Tensor& output) const
    {
        Op()->ConstantPad2D(*this, left, right, top, bottom, value, output);
    }

    //////////////////////////////////////////////////////////////////////////
    Tensor Tensor::ConstantPad2D(uint32_t left, uint32_t right, uint32_t top, uint32_t bottom, float value) const
    {
        Tensor output(Shape(Width() + left + right, Height() + top + bottom, Depth(), Batch()));
        ConstantPad2D(left, right, top, bottom, value, output);
        return output;
    }

    //////////////////////////////////////////////////////////////////////////
    void Tensor::ReflectPad2D(uint32_t left, uint32_t right, uint32_t top, uint32_t bottom, Tensor& output) const
    {
        Op()->ReflectPad2D(*this, left, right, top, bottom, output);
    }

    //////////////////////////////////////////////////////////////////////////
    Tensor Tensor::ReflectPad2D(uint32_t left, uint32_t right, uint32_t top, uint32_t bottom) const
    {
        Tensor output(Shape(Width() + left + right, Height() + top + bottom, Depth(), Batch()));
        ReflectPad2D(left, right, top, bottom, output);
        return output;
    }

    //////////////////////////////////////////////////////////////////////////
    void Tensor::Pad2DGradient(const Tensor& gradient, uint32_t left, uint32_t right, uint32_t top, uint32_t bottom, Tensor& inputsGradient) const
    {
        Op()->Pad2DGradient(gradient, left, right, top, bottom, inputsGradient);
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
        else if (axis == _01Axes)
            maxIndex.Resize(Shape(1, 1, Len(2), Len(3)));
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
        else if (axis == _01Axes)
            minIndex.Resize(Shape(1, 1, Len(2), Len(3)));
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
	Tensor Tensor::Transpose() const
	{
		Tensor output(Shape(Height(), Width(), Depth(), Batch()));
		Transpose(output);
		return output;
	}

    //////////////////////////////////////////////////////////////////////////
    Tensor Tensor::Transpose(const vector<EAxis>& permutation) const
    {
        Tensor output(Shape(m_Shape.Dimensions[permutation[0]], m_Shape.Dimensions[permutation[1]], m_Shape.Dimensions[permutation[2]], m_Shape.Dimensions[permutation[3]]));
        Transpose(permutation, output);
        return output;
    }

    //////////////////////////////////////////////////////////////////////////
    void Tensor::Transpose(const vector<EAxis>& permutation, Tensor& output) const
    {
        NEURO_ASSERT(permutation.size() == 4, "Invalid number of axes in permutation, expected 4, found " << permutation.size());
        Op()->Transpose(*this, permutation, output);
    }

    //////////////////////////////////////////////////////////////////////////
	void Tensor::Transpose(Tensor& output) const
	{
		Op()->Transpose(*this, output);
	}

    //////////////////////////////////////////////////////////////////////////
	Tensor Tensor::Reshaped(const Shape& shape) const
	{
        CopyToHost();
        Tensor result(*this);
        result.Reshape(shape);
		return result;
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
        m_Storage.Resize(shape.Length);
    }

    //////////////////////////////////////////////////////////////////////////
    void Tensor::Resize(uint32_t length)
    {
        NEURO_ASSERT(m_Shape.Width() == m_Shape.Length, "");
        Resize(Shape(length));
    }

    //////////////////////////////////////////////////////////////////////////
    void Tensor::ResizeBatch(uint32_t batch)
    {
        if (Batch() == batch)
            return;

        Resize(Shape::From(m_Shape, batch));
    }

    //////////////////////////////////////////////////////////////////////////
    float Tensor::Norm() const
    {
        return ::sqrt(Pow(2).Sum(GlobalAxis)(0));
    }

    //////////////////////////////////////////////////////////////////////////
	Tensor Tensor::Resized(uint32_t width, uint32_t height, uint32_t depth) const
	{
		uint32_t newBatchLength = width * height * depth;
		Tensor result(Shape(width, height, depth, m_Shape.Batch()));
		for (uint32_t n = 0; n < Batch(); ++n)
		for (uint32_t i = 0, idx = n * newBatchLength; i < newBatchLength; ++i, ++idx)
			result.m_Storage.Data()[idx] = m_Storage.Data()[n * BatchLength() + i % BatchLength()];
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
    void Tensor::Conv2D(const Tensor& kernels, uint32_t stride, uint32_t padding, EDataFormat dataFormat, Tensor& output) const
	{
        assert((dataFormat == NCHW ? Depth() : Len(0)) == kernels.Depth());
		Op()->Conv2D(*this, kernels, stride, padding, padding, dataFormat, output);
	}

	//////////////////////////////////////////////////////////////////////////
    Tensor Tensor::Conv2D(const Tensor& kernels, uint32_t stride, uint32_t padding, EDataFormat dataFormat) const
	{
		Tensor output(GetConvOutputShape(GetShape(), kernels.Batch(), kernels.Width(), kernels.Height(), stride, padding, padding, dataFormat));
		Conv2D(kernels, stride, padding, dataFormat, output);
		return output;
	}

    //////////////////////////////////////////////////////////////////////////
    void Tensor::Conv2DBiasActivation(const Tensor& kernels, uint32_t stride, uint32_t padding, const Tensor& bias, EActivation activation, float activationAlpha, Tensor& output) const
    {
        Op()->Conv2DBiasActivation(*this, kernels, stride, padding, padding, bias, activation, activationAlpha, output);
    }

    //////////////////////////////////////////////////////////////////////////
    Tensor Tensor::Conv2DBiasActivation(const Tensor& kernels, uint32_t stride, uint32_t padding, const Tensor& bias, EActivation activation, float activationAlpha) const
    {
        Tensor output(GetConvOutputShape(GetShape(), kernels.Batch(), kernels.Width(), kernels.Height(), stride, padding, padding, NCHW));
        Conv2DBiasActivation(kernels, stride, padding, bias, activation, activationAlpha, output);
        return output;
    }

    //////////////////////////////////////////////////////////////////////////
    void Tensor::Conv2DBiasGradient(const Tensor& gradient, Tensor& inputsGradient) const
    {
        Op()->Conv2DBiasGradient(gradient, inputsGradient);
    }

    //////////////////////////////////////////////////////////////////////////
	void Tensor::Conv2DInputsGradient(const Tensor& gradient, const Tensor& kernels, uint32_t stride, uint32_t padding, EDataFormat dataFormat, Tensor& inputsGradient) const
	{
		Op()->Conv2DInputGradient(gradient, kernels, stride, padding, padding, dataFormat, inputsGradient);
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::Conv2DKernelsGradient(const Tensor& input, const Tensor& gradient, uint32_t stride, uint32_t padding, EDataFormat dataFormat, Tensor& kernelsGradient) const
	{
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
        gradient.Conv2D(kernels, stride, padding, dataFormat, inputsGradient);
    }

    //////////////////////////////////////////////////////////////////////////
    void Tensor::Conv2DTransposedKernelsGradient(const Tensor& input, const Tensor& gradient, uint32_t stride, uint32_t padding, EDataFormat dataFormat, Tensor& kernelsGradient) const
    {
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
		Tensor result(GetPooling2DOutputShape(GetShape(), filterSize, filterSize, stride, padding, padding, dataFormat));
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
    void Tensor::BatchNorm(const Tensor& gamma, const Tensor& beta, float epsilon, const Tensor* runningMean, const Tensor* runningVar, Tensor& result) const
    {
        NEURO_ASSERT((runningMean && runningVar) || (!runningMean && !runningVar), "Both running mean and var must be present or absent at the same time.");
        Op()->BatchNormalization(*this, m_Shape.Depth() > 1 ? Spatial : PerActivation, gamma, beta, epsilon, runningMean, runningVar, result);
    }

    //////////////////////////////////////////////////////////////////////////
    void Tensor::BatchNormTrain(const Tensor& gamma, const Tensor& beta, float momentum, float epsilon, Tensor* runningMean, Tensor* runningVar, Tensor& saveMean, Tensor& saveInvVariance, Tensor& result) const
    {
        NEURO_ASSERT((runningMean && runningVar) || (!runningMean && !runningVar), "Both running mean and var must be present or absent at the same time.");
        Op()->BatchNormalizationTrain(*this, m_Shape.Depth() > 1 ? Spatial : PerActivation, gamma, beta, momentum, epsilon, runningMean, runningVar, saveMean, saveInvVariance, result);
    }

    //////////////////////////////////////////////////////////////////////////
    void Tensor::BatchNormGradient(const Tensor& input, const Tensor& gamma, float epsilon, const Tensor& outputGradient, const Tensor& savedMean, const Tensor& savedInvVariance, Tensor& gammaGradient, Tensor& betaGradient, bool trainable, Tensor& inputGradient) const
    {
        Op()->BatchNormalizationGradient(input, input.m_Shape.Depth() > 1 ? Spatial : PerActivation, gamma, epsilon, outputGradient, savedMean, savedInvVariance, gammaGradient, betaGradient, trainable, inputGradient);
    }

    //////////////////////////////////////////////////////////////////////////
    void Tensor::InstanceNorm(const Tensor& gamma, const Tensor& beta, float epsilon, Tensor& result) const
    {
        Op()->BatchNormalization(*this, Instance, gamma, beta, epsilon, nullptr, nullptr, result);
    }

    //////////////////////////////////////////////////////////////////////////
    void Tensor::InstanceNormTrain(const Tensor& gamma, const Tensor& beta, float epsilon, Tensor& saveMean, Tensor& saveInvVariance, Tensor& result) const
    {
        Op()->BatchNormalizationTrain(*this, Instance, gamma, beta, 1.f, epsilon, nullptr, nullptr, saveMean, saveInvVariance, result);
    }

    //////////////////////////////////////////////////////////////////////////
    void Tensor::InstanceNormGradient(const Tensor& input, const Tensor& gamma, float epsilon, const Tensor& outputGradient, const Tensor& savedMean, const Tensor& savedInvVariance, Tensor& gammaGradient, Tensor& betaGradient, bool trainable, Tensor& inputGradient) const
    {
        Op()->BatchNormalizationGradient(input, Instance, gamma, epsilon, outputGradient, savedMean, savedInvVariance, gammaGradient, betaGradient, trainable, inputGradient);
    }

    //////////////////////////////////////////////////////////////////////////
    void Tensor::Dropout(float prob, Tensor& saveMask, Tensor& output) const
    {
        Op()->Dropout(*this, prob, saveMask, output);
    }

    //////////////////////////////////////////////////////////////////////////
    void Tensor::DropoutGradient(const Tensor& outputGradient, float prob, const Tensor& savedMask, Tensor& inputGradient) const
    {
        Op()->DropoutGradient(outputGradient, prob, savedMask, inputGradient);
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
    bool Tensor::SameDimensionsOrOne(const Tensor& t) const
    {
        return (t.Width() == 1 || Width() == t.Width()) &&
               (t.Height() == 1 || Height() == t.Height()) &&
               (t.Depth() == 1 || Depth() == t.Depth()) &&
               (t.Batch() == 1 || Batch() == t.Batch());
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
        {
            NEURO_ASSERT((inputShape.Width() + 2 * paddingX - kernelWidth) >= 0, "");
            NEURO_ASSERT((inputShape.Height() + 2 * paddingY - kernelHeight) >= 0, "");
            return Shape((int)floor((inputShape.Width() + 2 * paddingX - kernelWidth) / (float)stride) + 1, 
                         (int)floor((inputShape.Height() + 2 * paddingY - kernelHeight) / (float)stride) + 1,
                         inputShape.Depth(),
                         inputShape.Batch());
        }

        NEURO_ASSERT((inputShape.Len(1) + 2 * paddingX - kernelWidth) >= 0, "");
        NEURO_ASSERT((inputShape.Len(2) + 2 * paddingY - kernelHeight) >= 0, "");
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
        {
            NEURO_ASSERT((inputShape.Width() + 2 * paddingX - kernelWidth) >= 0, "");
            NEURO_ASSERT((inputShape.Height() + 2 * paddingY - kernelHeight) >= 0, "");
            return Shape((int)floor((inputShape.Width() + 2 * paddingX - kernelWidth) / (float)stride) + 1, 
                         (int)floor((inputShape.Height() + 2 * paddingY - kernelHeight) / (float)stride) + 1,
                         kernelsNum,
                         inputShape.Batch());
        }

        NEURO_ASSERT((inputShape.Len(1) + 2 * paddingX - kernelWidth) >= 0, "");
        NEURO_ASSERT((inputShape.Len(2) + 2 * paddingY - kernelHeight) >= 0, "");
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
        {
            NEURO_ASSERT(((inputShape.Width() - 1) * stride + kernelWidth - 2 * paddingX) > 0, "");
            NEURO_ASSERT(((inputShape.Height() - 1) * stride + kernelHeight - 2 * paddingY) > 0, "");
            return Shape((inputShape.Width() - 1) * stride + kernelWidth - 2 * paddingX,
                         (inputShape.Height() - 1) * stride + kernelHeight - 2 * paddingY,
                         outputDepth, 
                         inputShape.Batch());
        }

        NEURO_ASSERT(((inputShape.Len(1) - 1) * stride + kernelWidth - 2 * paddingX) > 0, "");
        NEURO_ASSERT(((inputShape.Len(2) - 1) * stride + kernelHeight - 2 * paddingY) > 0, "");
        return Shape(outputDepth, 
                     (inputShape.Len(1) - 1) * stride + kernelWidth - 2 * paddingX,
                     (inputShape.Len(2) - 1) * stride + kernelHeight - 2 * paddingY,
                     inputShape.Batch());
    }

    void Tensor::SaveBin(ostream& stream) const
    {
        /*m_Shape.SaveBin(stream);
        size_t valuesNum = m_Storage.UsedSize();
        stream.write((const char*)&valuesNum, sizeof(valuesNum));
        stream.write((const char*)&m_Storage.Data()[0], valuesNum * sizeof(float));
        size_t nameLen = m_Name.length();
        stream.write((const char*)&nameLen, sizeof(nameLen));
        stream.write(m_Name.c_str(), nameLen);*/
    }

    void Tensor::LoadBin(istream& stream)
    {
        /*m_Shape.LoadBin(stream);
        m_Storage.Resize(m_Shape.Length);
        size_t valuesNum = 0;
        stream.read((char*)&valuesNum, sizeof(valuesNum));
        stream.read((char*)m_Storage.Data(), valuesNum * sizeof(float));
        size_t nameLen = 0;
        stream.read((char*)&nameLen, sizeof(nameLen));
        m_Name.resize(nameLen);
        stream.read(&m_Name[0], nameLen);*/
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
	void Tensor::CopyTo(Tensor& target, float tau) const
	{
        NEURO_ASSERT(m_Shape.Length == target.m_Shape.Length, "");

        if (tau <= 0 && (target.IsOnDevice() || IsOnDevice())) // device is more important
        {
            CopyToDevice();
            target.OverrideDevice();

            m_Storage.CopyWithinDevice(target.GetDevicePtr());
            return;
        }

		CopyToHost();
        target.OverrideHost();

		if (tau <= 0)
            memcpy(target.m_Storage.Data(), m_Storage.Data(), m_Storage.SizeInBytes());
		else
			Map([&](float v1, float v2) { return v1 * tau + v2 * (1 - tau); }, target, target);
	}

    //////////////////////////////////////////////////////////////////////////
    void Tensor::CopyTo(size_t offset, Tensor& target, size_t targetOffset, size_t elementsNum) const
    {
        if (target.IsOnDevice()) // target is more important
        {
            CopyToDevice();
            m_Storage.CopyWithinDevice(target.GetDevicePtr() + targetOffset, GetDevicePtr() + offset, elementsNum * sizeof(float));
            return;
        }

        CopyToHost();
        target.CopyToHost(true);

        memcpy(target.m_Storage.Data() + targetOffset, m_Storage.Data() + offset, elementsNum * sizeof(float));
    }

	//////////////////////////////////////////////////////////////////////////
	void Tensor::CopyBatchTo(uint32_t batchId, uint32_t targetBatchId, Tensor& target) const
	{
        NEURO_ASSERT(SameDimensionsExceptBatches(target), "");
        NEURO_ASSERT(batchId < Batch(), "");
        NEURO_ASSERT(targetBatchId < target.Batch(), "");
		
        CopyTo(batchId * m_Shape.Dim0Dim1Dim2, target, targetBatchId * m_Shape.Dim0Dim1Dim2, m_Shape.Dim0Dim1Dim2);
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::CopyDepthTo(uint32_t depthId, uint32_t batchId, uint32_t targetDepthId, uint32_t targetBatchId, Tensor& target) const
	{
        NEURO_ASSERT(Width() == target.Width() && Height() == target.Height(), "Incompatible tensors.");

        CopyTo(batchId * m_Shape.Dim0Dim1Dim2 + depthId * m_Shape.Dim0Dim1, target, targetBatchId * m_Shape.Dim0Dim1Dim2 + targetDepthId * m_Shape.Dim0Dim1, m_Shape.Dim0Dim1);
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
		if (m_Storage.Size() != other.m_Storage.Size())
			return false;

		if (epsilon == 0)
            return memcmp(m_Storage.Data(), other.m_Storage.Data(), m_Storage.SizeInBytes()) == 0;

		for (uint32_t i = 0; i < m_Storage.Size(); ++i)
			if (abs(m_Storage.Data()[i] - other.m_Storage.Data()[i]) > epsilon)
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

        auto inputValues = input.Values();
        auto minValueValues = maxValue.Values();

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
                        maxIndex->Values()[idx] = (float)inputShape.GetIndex(w, h, d, n);
                    else if (W && H && D)
                        maxIndex->Values()[idx] = (float)inputShape.GetIndex(w, h, d, 0u);
                    else
                        maxIndex->Values()[idx] = (float)(w * W + h * H + d * D + n * N);
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
        if (axis == _01Axes)
            return MaxTemplate<1, 1, 0, 0>(*this, maxIndex);
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

        auto inputValues = input.Values();
        auto minValueValues = minValue.Values();

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
                        minIndex->Values()[idx] = (float)inputShape.GetIndex(w, h, d, n);
                    else if (W && H && D)
                        minIndex->Values()[idx] = (float)inputShape.GetIndex(w, h, d, 0u);
                    else
                        minIndex->Values()[idx] = (float)(w * W + h * H + d * D + n * N);
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
        if (axis == _01Axes)
            return MinTemplate<1, 1, 0, 0>(*this, minIndex);
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
    void Tensor::Activation(EActivation activation, float coeff, Tensor& output) const
    {
        switch (activation)
        {
        case _Sigmoid:
            return Sigmoid(output);
        case _ReLU:
            return ReLU(output);
        case _TanH:
            return Tanh(output);
        case _ELU:
            return Elu(coeff, output);
        case _LeakyReLU:
            return LeakyReLU(coeff, output);
        case _Softmax:
            return Softmax(output);
        }
        NEURO_ASSERT(false, "Unsupported activation.");
    }

    //////////////////////////////////////////////////////////////////////////
    void Tensor::ActivationGradient(EActivation activation, float coeff, const Tensor& output, const Tensor& outputGradient, Tensor& inputGradient) const
    {
        switch (activation)
        {
        case _Sigmoid:
            return SigmoidGradient(output, outputGradient, inputGradient);
        case _ReLU:
            return ReLUGradient(output, outputGradient, inputGradient);
        case _TanH:
            return TanhGradient(output, outputGradient, inputGradient);
        case _ELU:
            return EluGradient(output, outputGradient, coeff, inputGradient);
        case _LeakyReLU:
            return LeakyReLUGradient(output, outputGradient, coeff, inputGradient);
        case _Softmax:
            return SoftmaxGradient(output, outputGradient, inputGradient);
        }
        NEURO_ASSERT(false, "Unsupported activation.");
    }

    //////////////////////////////////////////////////////////////////////////
    void Tensor::Sigmoid(Tensor& result) const
    {
        Op()->Sigmoid(*this, result);
    }

    //////////////////////////////////////////////////////////////////////////
    void Tensor::SigmoidGradient(const Tensor& output, const Tensor& outputGradient, Tensor& inputGradient) const
    {
        Op()->SigmoidGradient(output, outputGradient, inputGradient);
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
    void Tensor::ReLUGradient(const Tensor& output, const Tensor& outputGradient, Tensor& inputGradient) const
    {
        Op()->ReLUGradient(output, outputGradient, inputGradient);
    }

	//////////////////////////////////////////////////////////////////////////
	void Tensor::Elu(float alpha, Tensor& result) const
	{
		Op()->Elu(*this, alpha, result);
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::EluGradient(const Tensor& output, const Tensor& outputGradient, float alpha, Tensor& inputGradient) const
	{
		Op()->EluGradient(output, outputGradient, alpha, inputGradient);
	}

    //////////////////////////////////////////////////////////////////////////
    void Tensor::LeakyReLU(float alpha, Tensor& result) const
    {
        Op()->LeakyReLU(*this, alpha, result);
    }

    //////////////////////////////////////////////////////////////////////////
    void Tensor::LeakyReLUGradient(const Tensor& output, const Tensor& outputGradient, float alpha, Tensor& inputGradient) const
    {
        Op()->LeakyReLUGradient(output, outputGradient, alpha, inputGradient);
    }

	//////////////////////////////////////////////////////////////////////////
	void Tensor::Softmax(Tensor& result) const
	{
		Op()->Softmax(*this, result);
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::SoftmaxGradient(const Tensor& output, const Tensor& outputGradient, Tensor& inputGradient) const
	{
		Op()->SoftmaxGradient(output, outputGradient, inputGradient);
	}

	//////////////////////////////////////////////////////////////////////////
	void Tensor::CopyToDevice() const
	{
        m_Storage.CopyToDevice();
	}

    //////////////////////////////////////////////////////////////////////////
	void Tensor::CopyToHost(bool allowAlloc) const
	{
        m_Storage.CopyToHost(allowAlloc);
	}

    //////////////////////////////////////////////////////////////////////////
    void Tensor::SyncToHost() const
    {
        m_Storage.SyncToHost();
    }

    //////////////////////////////////////////////////////////////////////////
    bool Tensor::TryDeviceAllocate() const
    {
        if (!m_Storage.IsHostAllocated())
            m_Storage.AllocateOnHost();
        if (Op() == g_OpGpu)
        {
            m_Storage.AllocateOnDevice();
            return m_Storage.IsDeviceAllocated();
        }
        return false;
    }

    //////////////////////////////////////////////////////////////////////////
    bool Tensor::TryDeviceRelease()
    {
        if (Op() != g_OpGpu)
            return false;

        m_Storage.FreeOnDevice();
        return true;
    }

    //////////////////////////////////////////////////////////////////////////
    void Tensor::Prefetch() const
    {
        if (Op() != g_OpGpu)
            return;

        m_Storage.Preload();
    }

    //////////////////////////////////////////////////////////////////////////
    /*void Tensor::ScheduleOffload() const
    {
        if (Op() != g_OpGpu)
            return;

        m_Storage.ScheduleOffload();
    }*/

    //////////////////////////////////////////////////////////////////////////
    void Tensor::Offload(bool force) const
    {
        if (Op() != g_OpGpu)
            return;

        m_Storage.Offload(force);
    }

    //////////////////////////////////////////////////////////////////////////
    void Tensor::ResetDeviceRef(size_t n)
    {
        m_Storage.ResetDeviceRef(n);
    }

    //////////////////////////////////////////////////////////////////////////
    void Tensor::IncDeviceRef(size_t n)
    {
        m_Storage.IncDeviceRef(n);
    }

    //////////////////////////////////////////////////////////////////////////
    void Tensor::DecDeviceRef(size_t n)
    {
        m_Storage.DecDeviceRef(n);
    }

    //////////////////////////////////////////////////////////////////////////
    void Tensor::ResetRef(size_t n)
    {
        m_Storage.ResetRef(n);
    }

    //////////////////////////////////////////////////////////////////////////
    void Tensor::IncRef(size_t n)
    {
        m_Storage.IncRef(n);
    }

    //////////////////////////////////////////////////////////////////////////
    void Tensor::DecRef(size_t n)
    {
        m_Storage.DecRef(n);
    }

    //////////////////////////////////////////////////////////////////////////
    void Tensor::ReleaseData()
    {
        m_Storage.Release();
    }

    //////////////////////////////////////////////////////////////////////////
    void Tensor::OverrideHost()
    {
        m_Storage.OverrideHost();
    }

    //////////////////////////////////////////////////////////////////////////
    void Tensor::OverrideDevice()
    {
        m_Storage.OverrideDevice();
    }

    //////////////////////////////////////////////////////////////////////////
    const float* Tensor::GetDevicePtr() const
    {
        return m_Storage.DeviceData();
    }

    //////////////////////////////////////////////////////////////////////////
    float* Tensor::GetDevicePtr()
    {
        return m_Storage.DeviceData();
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
        if (!m_Storage.AllocSizeInBytes() || !m_Storage.IsHostAllocated())
            return;

        SyncToHost();
        ofstream stream(Replace(outFile, "/", "-"));
        for (int i = 0; i < 4; ++i)
            stream << m_Shape.Dimensions[i] << "\n";
        stream << fixed << setprecision(6);
        for (uint32_t i = 0; i < m_Shape.Length; ++i)
            stream << m_Storage.DataUnsafe()[i] << "\n";
        stream.close();
    }

    //////////////////////////////////////////////////////////////////////////
    void Tensor::DebugRecoverValues(const string& inFile)
    {
        ifstream stream(inFile);

        if (!stream)
        {
            NEURO_ASSERT(false, "File '" << inFile << "' not found.");
            return;
        }

        OverrideHost();
        vector<int> dimensions(4);
        for (int i = 0; i < 4; ++i)
            stream >> dimensions[i];
        m_Shape = Shape::From(dimensions);
        m_Storage.Resize(m_Shape.Length);
        for (uint32_t i = 0; i < m_Shape.Length; ++i)
            stream >> m_Storage.Data()[i];
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
    Tensor operator/(const Tensor& t1, const Tensor& t2)
    {
        return t1.Div(t2);
    }

    //////////////////////////////////////////////////////////////////////////
    Tensor operator/(const Tensor& t, float v)
    {
        return t.Div(v);
    }

    //////////////////////////////////////////////////////////////////////////
    Tensor operator/(float v, const Tensor& t)
    {
        return t.Inversed(v);
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
        return t.Pow(p);
    }

    //////////////////////////////////////////////////////////////////////////
    Tensor sqr(const Tensor& t)
    {
        return t.Pow(2);
    }

    //////////////////////////////////////////////////////////////////////////
    Tensor sqrt(const Tensor& t)
    {
        return t.Sqrt();
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
