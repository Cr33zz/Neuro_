#pragma once

#include <string>
#include <vector>
#include "assert.h"

namespace Neuro
{
	using namespace std;

    class Shape
    {
	public:
        explicit Shape(int width = 1, int height = 1, int depth = 1, int batchSize = 1);

		bool operator==(const Shape& other) const;
		bool operator!=(const Shape& other) const;

        static Shape From(const vector<int>& dimensions);
        static Shape From(const int* dimensions, int dimNum);
        static Shape From(const Shape& shapeWithoutBatches, int batchSize);

		static const int Auto = -1; // Automatically guesses

        Shape Reshaped(int w, int h, int d, int n) const;
        int GetIndex(int w, int h = 0, int d = 0, int n = 0) const;

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

		int Width() const { return Dimensions[0]; }
		int Height() const { return Dimensions[1]; }
		int Depth() const { return Dimensions[2]; }
		int Batch() const { return Dimensions[3]; }

		int Dimensions[4];

		int Dim0;
        int Dim0Dim1;
        int Dim0Dim1Dim2;
		int Length;
    };
}
