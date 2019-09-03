#pragma once

#include <string>
#include <vector>
#include <assert.h>

#include "Types.h"

namespace Neuro
{
	using namespace std;

    class Shape
    {
	public:
        explicit Shape(uint width = 1, uint height = 1, uint depth = 1, uint batchSize = 1);

		bool operator==(const Shape& other) const;
		bool operator!=(const Shape& other) const;

        static Shape From(const vector<int>& dimensions);
        static Shape From(const int* dimensions, int dimNum);
        static Shape From(const Shape& shapeWithoutBatches, int batchSize);

		static const int Auto = -1; // Automatically guesses

        // One of the provided dimensions can be -1. In that case it will be guessed based on the length and other dimensions.
        Shape Reshaped(int w, int h, int d, int n) const;

        uint GetIndex(uint w, uint h = 0, uint d = 0, uint n = 0) const;
        uint GetIndex(int w, int h = 0, int d = 0, int n = 0) const;
        uint GetIndex(const vector<uint>& indices) const;
        uint GetIndexNCHW(const vector<uint>& indices) const;

		string ToString() const;

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

        uint Width() const { return Dimensions[0]; }
        uint Height() const { return Dimensions[1]; }
        uint Depth() const { return Dimensions[2]; }
        uint Batch() const { return Dimensions[3]; }

        uint Dimensions[4];

        uint Dim0;
        uint Dim0Dim1;
        uint Dim0Dim1Dim2;
        uint Length;
        uint NDim;
    };
}
