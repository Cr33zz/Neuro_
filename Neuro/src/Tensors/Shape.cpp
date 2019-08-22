#include <sstream>
#include "assert.h"

#include "Tensors/Shape.h"

namespace Neuro
{
	//////////////////////////////////////////////////////////////////////////
	Shape::Shape(int width, int height, int depth, int batchSize)
		: Dim0(width), Dim0Dim1(Dim0 * height), Dim0Dim1Dim2(Dim0Dim1 * depth), Length(Dim0Dim1Dim2 * batchSize)
	{
		Dimensions[0] = width;
		Dimensions[1] = height;
		Dimensions[2] = depth;
		Dimensions[3] = batchSize;
	}

	//////////////////////////////////////////////////////////////////////////
	Neuro::Shape Shape::From(const vector<int>& dimensions)
	{
		return From(dimensions.data(), (int)dimensions.size());
	}

	//////////////////////////////////////////////////////////////////////////
	Neuro::Shape Shape::From(const int* dimensions, int dimNum)
	{
		switch (dimNum)
		{
		case 1: return Shape(dimensions[0]);
		case 2: return Shape(dimensions[0], dimensions[1]);
		case 3: return Shape(dimensions[0], dimensions[1], dimensions[2]);
		default: return Shape(dimensions[0], dimensions[1], dimensions[2], dimensions[3]);
		}
	}

    //////////////////////////////////////////////////////////////////////////
    Neuro::Shape Shape::From(const Shape& shapeWithoutBatches, int batchSize)
    {
        return Shape(shapeWithoutBatches.Width(), shapeWithoutBatches.Height(), shapeWithoutBatches.Depth(), batchSize);
    }

    //////////////////////////////////////////////////////////////////////////
	Neuro::Shape Shape::Reshaped(int w, int h, int d, int n) const
	{
		int dimensions[4] = { w, h, d, n };
		int dToUpdate = -1;
		int product = 1;
		for (int d = 0; d < 4; ++d)
		{
			if (dimensions[d] == -1)
			{
				dToUpdate = d;
				continue;
			}

			product *= dimensions[d];
		}

		if (dToUpdate >= 0)
		{
			dimensions[dToUpdate] = Length / product;
		}

		return From(dimensions, 4);
	}

	//////////////////////////////////////////////////////////////////////////
	int Shape::GetIndex(int w, int h /*= 0*/, int d /*= 0*/, int n /*= 0*/) const
	{
		assert(w < Width());
		assert(h < Height());
		assert(d < Depth());
		assert(n < Batch());
		return Dim0Dim1Dim2 * n + Dim0Dim1 * d + Dim0 * h + w;
	}

	//////////////////////////////////////////////////////////////////////////
	std::string Shape::ToString() const
	{
		stringstream ss;
		ss << Width() << "x" << Height() << "x" << Depth() << "x" << Batch();
		return ss.str();
	}

	//////////////////////////////////////////////////////////////////////////
	bool Shape::operator!=(const Shape& other) const
	{
		return !(*this == other);
	}

	//////////////////////////////////////////////////////////////////////////
	bool Shape::operator==(const Shape& other) const
	{
		if (Length != other.Length)
			return false;

		return Width() == other.Width() && Height() == other.Height() && Depth() == other.Depth();
	}

}