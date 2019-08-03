#pragma once

#include <string>
#include <sstream>
#include <vector>
#include <array>
#include "assert.h"

namespace Neuro
{
	using namespace std;

    class Shape
    {
	public:
        Shape(int width = 1, int height = 1, int depth = 1, int batchSize = 1)
			: Dim0(width), Dim0Dim1(Dim0 * height), Dim0Dim1Dim2(Dim0Dim1 * depth), Length(Dim0Dim1Dim2 * batchSize)
        {
			Dimensions[0] = width;
			Dimensions[1] = height;
			Dimensions[2] = depth;
			Dimensions[3] = batchSize;
		}

		bool operator==(const Shape& other) const
		{
			if (Length != other.Length)
				return false;

			return Width() == other.Width() && Height() == other.Height() && Depth() == other.Depth();
		}

		bool operator!=(const Shape& other) const
		{
			return !(*this == other);
		}

        static Shape From(const vector<int>& dimensions)
        {
            return From(dimensions.data(), (int)dimensions.size());
        }

        static Shape From(const int* dimensions, int dimNum)
        {
            switch (dimNum)
            {
            case 1: return Shape(dimensions[0]);
            case 2: return Shape(dimensions[0], dimensions[1]);
            case 3: return Shape(dimensions[0], dimensions[1], dimensions[2]);
            default: return Shape(dimensions[0], dimensions[1], dimensions[2], dimensions[3]);
            }
        }

		static const int Auto = -1; // Automatically guesses

        Shape Reshaped(int w, int h, int d, int n) const
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

        int GetIndex(int w, int h = 0, int d = 0, int n = 0) const
        {
            assert(w < Width());
			assert(h < Height());
			assert(d < Depth());
			assert(n < BatchSize());
            return Dim0Dim1Dim2 * n + Dim0Dim1 * d + Dim0 * h + w;
        }

		string ToString() const
		{
			stringstream ss;
			ss << Width() << "x" << Height() << "x" << Depth() << "x" << BatchSize();
			return ss.str();
		}

        /*void Serialize(BinaryWriter writer)
        {
            writer.Write(Width);
            writer.Write(Height);
            writer.Write(Depth);
            writer.Write(BatchSize);
        }

        static Shape Deserialize(BinaryReader reader)
        {
            int[] dims = new [] { reader.ReadInt32(), reader.ReadInt32(), reader.ReadInt32(), reader.ReadInt32() };
            return Shape.From(dims);
        }*/

		int Width() const { return Dimensions[0]; }
		int Height() const { return Dimensions[1]; }
		int Depth() const { return Dimensions[2]; }
		int BatchSize() const { return Dimensions[3]; }

		int Dimensions[4];

		int Dim0;
        int Dim0Dim1;
        int Dim0Dim1Dim2;
		int Length;
    };
}
