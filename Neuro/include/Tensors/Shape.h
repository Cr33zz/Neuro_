﻿#pragma once

#include <string>
#include <vector>
#include <assert.h>

#include "Types.h"

typedef struct cudnnTensorStruct* cudnnTensorDescriptor_t;

namespace Neuro
{
	using namespace std;    

    class NEURO_DLL_EXPORT Shape
    {
	public:
        Shape(uint32_t width = 0, uint32_t height = 1, uint32_t depth = 1, uint32_t batchSize = 1);
        Shape(istream& stream);
        Shape(const Shape& other);
        Shape(Shape&& other);
        Shape& operator=(const Shape& other);
        Shape& operator=(Shape&& other);
        ~Shape();

        bool IsValid() const { return NDim > 0; }

		bool operator==(const Shape& other) const;
		bool operator!=(const Shape& other) const;

        bool EqualsIgnoreBatch(const Shape& other) const;

        static Shape From(const vector<int>& dimensions);
        static Shape From(const int* dimensions, int dimNum);
        static Shape FromKeras(const int* dimensions, int dimNum);
        static Shape From(const Shape& shapeWithoutBatches, int batchSize);

		static const int Auto = -1; // Automatically guesses

        // One of the provided dimensions can be -1. In that case it will be guessed based on the length and other dimensions.
        Shape Reshaped(int w, int h, int d, int n) const;
        Shape Transposed(const vector<EAxis>& axes) const;

        uint32_t GetIndex(uint32_t w, uint32_t h = 0, uint32_t d = 0, uint32_t n = 0) const;
        uint32_t GetIndex(int w, int h = 0, int d = 0, int n = 0) const;
        uint32_t GetIndex(const vector<uint32_t>& indices) const;
        uint32_t GetIndexKeras(const vector<int>& indices) const;
        vector<int> KerasDims() const;

		string ToString() const;

        void SaveBin(ostream& stream) const;
        void LoadBin(istream& stream);

        uint32_t Width() const { return Dimensions[0]; }
        uint32_t Height() const { return Dimensions[1]; }
        uint32_t Depth() const { return Dimensions[2]; }
        uint32_t Batch() const { return Dimensions[3]; }
        uint32_t Len(size_t dim) const { return Dimensions[dim]; }
        uint32_t Str(size_t dim) const { return Stride[dim]; }

        const cudnnTensorDescriptor_t DeviceDesc() const;

        uint32_t Dimensions[4];
        uint32_t Stride[4];
        uint32_t Dim0;
        uint32_t Dim0Dim1;
        uint32_t Dim0Dim1Dim2;
        uint32_t Length;
        uint32_t NDim;

    private:
        mutable cudnnTensorDescriptor_t CudnnDesc = nullptr;
    };

    //////////////////////////////////////////////////////////////////////////
    _inline uint32_t Shape::GetIndex(int w, int h, int d, int n) const
	{
#       ifdef DEBUG
        assert(w >= -(int)Width() && w < (int)Width());
		assert(h >= -(int)Height() && h < (int)Height());
		assert(d >= -(int)Depth() && d < (int)Depth());
		assert(n >= -(int)Batch() && n < (int)Batch());
#       endif

		return Dim0Dim1Dim2 * ((n >= 0 ? n : (n + Dimensions[3])) % (int)Batch()) + 
               Dim0Dim1 * ((d >= 0 ? d : (d + Dimensions[2])) % (int)Depth()) +
               Dim0 * ((h >= 0 ? h : (h + Dimensions[1])) % (int)Height()) +
               ((w >= 0 ? w : (w + Dimensions[0])) % (int)Width());
	}

    //////////////////////////////////////////////////////////////////////////
    _inline uint32_t Shape::GetIndex(uint32_t w, uint32_t h, uint32_t d, uint32_t n) const
	{
#       ifdef DEBUG
		assert(w < Width());
		assert(h < Height());
		assert(d < Depth());
		assert(n < Batch());
#       endif

		return Dim0Dim1Dim2 * n + Dim0Dim1 * d + Dim0 * h + w;
	}
}
