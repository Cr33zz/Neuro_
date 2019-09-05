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
        explicit Shape(uint32_t width = 1, uint32_t height = 1, uint32_t depth = 1, uint32_t batchSize = 1);

		bool operator==(const Shape& other) const;
		bool operator!=(const Shape& other) const;

        static Shape From(const vector<int>& dimensions);
        static Shape From(const int* dimensions, int dimNum);
        static Shape From(const Shape& shapeWithoutBatches, int batchSize);

		static const int Auto = -1; // Automatically guesses

        // One of the provided dimensions can be -1. In that case it will be guessed based on the length and other dimensions.
        Shape Reshaped(int w, int h, int d, int n) const;

        uint32_t GetIndex(uint32_t w, uint32_t h = 0, uint32_t d = 0, uint32_t n = 0) const;
        uint32_t GetIndex(int w, int h = 0, int d = 0, int n = 0) const;
        uint32_t GetIndex(const vector<uint32_t>& indices) const;
        uint32_t GetIndexNCHW(const vector<uint32_t>& indices) const;

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

        uint32_t Width() const { return Dimensions[0]; }
        uint32_t Height() const { return Dimensions[1]; }
        uint32_t Depth() const { return Dimensions[2]; }
        uint32_t Batch() const { return Dimensions[3]; }

        uint32_t Dimensions[4];

        uint32_t Dim0;
        uint32_t Dim0Dim1;
        uint32_t Dim0Dim1Dim2;
        uint32_t Length;
        uint32_t NDim;
    };
}
