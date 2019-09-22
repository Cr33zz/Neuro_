#include <sstream>
#include "assert.h"

#include "Tensors/Shape.h"

namespace Neuro
{
	//////////////////////////////////////////////////////////////////////////
    Shape::Shape(uint32_t width, uint32_t height, uint32_t depth, uint32_t batchSize)
		: Dim0(width), Dim0Dim1(Dim0 * height), Dim0Dim1Dim2(Dim0Dim1 * depth), Length(Dim0Dim1Dim2 * batchSize)
	{
		Dimensions[0] = width;
		Dimensions[1] = height;
		Dimensions[2] = depth;
		Dimensions[3] = batchSize;

        NDim = 0;
        if (batchSize > 1)
            NDim = 4;
        else if (depth > 1)
            NDim = 3;
        else if (height > 1)
            NDim = 2;
        else if (width > 0)
            NDim = 1;
	}

    //////////////////////////////////////////////////////////////////////////
    Shape::Shape(istream& stream)
    {
        LoadBin(stream);
    }

    //////////////////////////////////////////////////////////////////////////
    bool Shape::EqualsIgnoreBatch(const Shape& other) const
    {
        return Dimensions[0] == other.Dimensions[0] &&
               Dimensions[1] == other.Dimensions[1] &&
               Dimensions[2] == other.Dimensions[2];
    }

    //////////////////////////////////////////////////////////////////////////
	Shape Shape::From(const vector<int>& dimensions)
	{
		return From(dimensions.data(), (int)dimensions.size());
	}

	//////////////////////////////////////////////////////////////////////////
	Shape Shape::From(const int* dimensions, int dimNum)
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
    Shape Shape::From(const Shape& shapeWithoutBatches, int batchSize)
    {
        return Shape(shapeWithoutBatches.Width(), shapeWithoutBatches.Height(), shapeWithoutBatches.Depth(), batchSize);
    }

    //////////////////////////////////////////////////////////////////////////
	Shape Shape::Reshaped(int w, int h, int d, int n) const
	{
		int dimensions[4] = { w, h, d, n };
		int dToUpdate = -1;
        uint32_t product = 1;
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
			dimensions[dToUpdate] = (int)(Length / product);
            product = Length;
		}

        assert(Length == product);

		return From(dimensions, 4);
	}

	//////////////////////////////////////////////////////////////////////////
    uint32_t Shape::GetIndex(const vector<uint32_t>& indices) const
    {
        size_t indicesCount = indices.size();

        assert(indicesCount > 0);
        assert(indicesCount < 5);

        return GetIndex(
                indices[0], 
                indicesCount > 1 ? indices[1] : 0, 
                indicesCount > 2 ? indices[2] : 0, 
                indicesCount > 3 ? indices[3] : 0);
    }

    //////////////////////////////////////////////////////////////////////////
    uint32_t Shape::GetIndexNCHW(const vector<uint32_t>& indices) const
    {
        size_t indicesCount = indices.size();

        assert(indicesCount > 0);
        assert(indicesCount < 5);

        return GetIndex(
            indices.back(),
            indicesCount > 1 ? indices[indicesCount - 2] : 0,
            indicesCount > 2 ? indices[indicesCount - 3] : 0,
            indicesCount > 3 ? indices[0] : 0);
    }

    //////////////////////////////////////////////////////////////////////////
	std::string Shape::ToString() const
	{
		stringstream ss;
		ss << "(" << Width() << ", " << Height() << ", " << Depth() << ", " << Batch() << ")";
		return ss.str();
	}

    //////////////////////////////////////////////////////////////////////////
    void Shape::SaveBin(ostream& stream) const
    {
        stream.write((char*)Dimensions, sizeof(Dimensions));
        stream.write((char*)&Dim0, sizeof(Dim0));
        stream.write((char*)&Dim0Dim1, sizeof(Dim0Dim1));
        stream.write((char*)&Dim0Dim1Dim2, sizeof(Dim0Dim1Dim2));
        stream.write((char*)&Length, sizeof(Length));
        stream.write((char*)&NDim, sizeof(NDim));
    }

    //////////////////////////////////////////////////////////////////////////
    void Shape::LoadBin(istream& stream)
    {
        stream.read((char*)Dimensions, sizeof(Dimensions));
        stream.read((char*)&Dim0, sizeof(Dim0));
        stream.read((char*)&Dim0Dim1, sizeof(Dim0Dim1));
        stream.read((char*)&Dim0Dim1Dim2, sizeof(Dim0Dim1Dim2));
        stream.read((char*)&Length, sizeof(Length));
        stream.read((char*)&NDim, sizeof(NDim));
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