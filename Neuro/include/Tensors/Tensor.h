#pragma once

#include <functional>
#include <limits>
#include <string>
#include <sstream>
#include <vector>
#include "assert.h"
#include "Tensors/Shape.h"

namespace Neuro
{
	class TensorOpCpu;
	class Random;

	using namespace std;

    class Tensor
    {
	public:
		enum EOpMode
		{
			CPU,
			MultiCPU,
			GPU
		};

		enum ELocation
		{
			Host,
			Device
		};

		enum EPaddingType
		{
			Valid, // output matrix's size will be decreased (depending on kernel size)
			Same,  // output matrix's size will be the same (except for depth) as input matrix
			Full,  // output matrix's size will be increased (depending on kernel size)
		};

		enum class EPoolType
		{
			Max,
			Avg
		};

		Tensor();
        Tensor(const Shape& shape);
        Tensor(const vector<float>&, const Shape& shape);
        Tensor(const vector<float>&);
        Tensor(const Tensor& t);
        Tensor(string bmpFile, bool grayScale);

        void SetOpMode(EOpMode mode);

        int Width() const { return m_Shape.Width(); };
		int Height() const { return m_Shape.Height(); }
		int Depth() const { return m_Shape.Depth(); }
		int BatchSize() const { return m_Shape.BatchSize(); }
		int BatchLength() const { return m_Shape.Dim0Dim1Dim2; }
		int Length() const { return (int)Values.size(); }

        vector<float>& GetValues();

        /*Bitmap ToBitmap()
        {
            assert(BatchSize == 1);

            Bitmap output = new Bitmap(Width, Height);
            bool grayScale = (Depth == 1);

            for (int d = 0; d < Depth; ++d)
            for (int h = 0; h < Height; ++h)
            for (int w = 0; w < Width; ++w)
                output.SetPixel(w, h, grayScale ? Color.FromArgb((int)(Get(w, h) * 255), (int)(Get(w, h) * 255), (int)(Get(w, h) * 255))
                                                : Color.FromArgb((int)(Get(w, h) * 255), (int)(Get(w, h, 1) * 255), (int)(Get(w, h, 2) * 255)));

            return output;
        }*/

        Tensor& FillWithRand(int seed = -1, float min = -1, float max = 1);
        Tensor& FillWithRange(float start = 0, float increment = 1);
        Tensor& FillWithValue(float value);

        void Zero();
	
	private:
        void Mul(bool transposeT, const Tensor& t, Tensor& result);
        Tensor Mul(bool transposeT, const Tensor& t);

	public:
        void Mul(const Tensor& t, Tensor& result);
        Tensor Mul(const Tensor& t);
        void MulElem(const Tensor& t, Tensor& result);
        Tensor MulElem(const Tensor& t);
        void Mul(float v, Tensor& result);
        Tensor Mul(float v);
        void Div(const Tensor& t, Tensor& result);
        Tensor Div(const Tensor& t);
        void Div(float v, Tensor& result);
        Tensor Div(float v);
        void Add(float alpha, float beta, const Tensor& t, Tensor& result);
        void Add(const Tensor& t, Tensor& result);
        Tensor Add(const Tensor& t);
        Tensor Add(float alpha, float beta, const Tensor& t);
        void Add(float v, Tensor& result);
        Tensor Add(float v);
        void Sub(const Tensor& t, Tensor& result);
        Tensor Sub(const Tensor& t);
        void Sub(float v, Tensor& result);
        Tensor Sub(float v);
        void Negated(Tensor& result);
        Tensor Negated();
        void Clipped(float min, float max, Tensor& result);
        Tensor Clipped(float min, float max);
        Tensor DiagFlat();

        void Map(const function<float(float)>& func, Tensor& result);
		Tensor Map(const function<float(float)>& func);
		void Map(const function<float(float, float)>&, const Tensor& other, Tensor& result);
		Tensor Map(const function<float(float, float)>& func, const Tensor& other);

        Tensor SumBatches();
        float Sum(int batch = -1);
        Tensor SumPerBatch();

        Tensor AvgBatches();
		float Avg(int batch = -1);
        Tensor AvgPerBatch();

		float Max(int batch = -1);
        Tensor MaxPerBatch();

        static Tensor MergeIntoBatch(const vector<Tensor>& tensors);
        // In case number of tensors is smaller than forcedDepth, first tensor will be repeated to account for missing tensors
        static Tensor MergeIntoDepth(const vector<Tensor>& tensors, int forcedDepth = 0);

		// This operation will concatenate elements of all input tensors separately for each batch
		static void Concat(const vector<Tensor>& inputs, Tensor& result);
        // This is reverse Concat operation
        void Split(vector<Tensor>& outputs);

        static void MergeMin(const vector<Tensor>& inputs, Tensor& result);
        static void MergeMax(const vector<Tensor>& inputs, Tensor& result);
        static void MergeSum(const vector<Tensor>& inputs, Tensor& result);
        static void MergeAvg(const vector<Tensor>& inputs, Tensor& result);
        static void MergeMinMaxGradient(Tensor output, const vector<Tensor>& inputs, Tensor outputGradient, vector<Tensor>& results);
        static void MergeSumGradient(Tensor output, const vector<Tensor>& inputs, Tensor outputGradient, vector<Tensor>& results);
        static void MergeAvgGradient(Tensor output, const vector<Tensor>& inputs, Tensor outputGradient, vector<Tensor>& results);

        void Normalized(Tensor& result);
        // ArgMax will return local index inside given batch if batch is not -1
        int ArgMax(int batch = -1);
        Tensor ArgMaxPerBatch();
        const Tensor& transposed();
        void Transpose(Tensor& result);

        // Generates a new tensor with given dimensions and populate it with this tensor's values in index order.
        // One of dimensions can be -1, in that case it will be calculated based on remaining dimensions.
        Tensor Reshaped(Shape shape);
        void Reshape(Shape shape);

        // Create new tensor with different batch length and use current tensors values to fill the new tensor.
        // Number of batches will be the same as in source tensor.
        Tensor Resized(int width, int height = 1, int depth = 1);
        Tensor FlattenHoriz();
        Tensor FlattenVert();
        void Rotated180(Tensor& result);
        Tensor Rotated180();
        void Conv2D(Tensor kernels, int stride, EPaddingType padding, Tensor& result);
        Tensor Conv2D(Tensor kernels, int stride, EPaddingType padding);
        static void Conv2DInputsGradient(Tensor gradient, Tensor kernels, int stride, EPaddingType padding, Tensor inputsGradient);
        static void Conv2DKernelsGradient(Tensor input, Tensor gradient, int stride, EPaddingType padding, Tensor kernelsGradient);
        static void Conv2DGradient_old(Tensor input, Tensor kernels, Tensor gradient, int stride, EPaddingType padding, Tensor inputGradient, Tensor kernelsGradient);
        void Pool(int filterSize, int stride, EPoolType type, EPaddingType padding, Tensor& result);
        Tensor Pool(int filterSize, int stride, EPoolType type, EPaddingType padding);
        // Assuming result matrix is of the dimensions of input to pooling layer
        static void PoolGradient(Tensor output, Tensor input, Tensor outputGradient, int filterSize, int stride, EPoolType type, EPaddingType padding, Tensor& result);
        string ToString();
        bool SameDimensionsExceptBatches(const Tensor& t);

        static void GetPaddingParams(EPaddingType type, int width, int height, int kernelWidth, int kernelHeight, int stride, int& outHeight, int& outWidth, int& paddingX, int& paddingY);

        //internal void Serialize(XmlElement parentElem, string name)
        //{
        //    XmlElement elem = parentElem.OwnerDocument.CreateElement(name);
        //    XmlAttribute shapeAttrib = parentElem.OwnerDocument.CreateAttribute("shape");
        //    shapeAttrib.Value = string.Join(",", Shape.Dimensions);
        //    elem.Attributes.Append(shapeAttrib);
        //    elem.InnerText = string.Join(",", Values);
        //    //elem.InnerText = $"\n{this.ToString()}\n";
        //    parentElem.AppendChild(elem);
        //}

        //internal void Deserialize(XmlElement elem)
        //{
        //    Shape = Shape.From(elem.GetAttribute("shape").Split(',').Select(w => int.Parse(w)).ToArray());
        //    Values = elem.InnerText.Split(',').Select(w => float.Parse(w)).ToArray();
        //}

        //void Serialize(BinaryWriter writer)
        //{
        //    Shape.Serialize(writer);
        //    writer.Write(Values.Length);
        //    foreach (var val in Values)
        //        writer.Write(val);
        //}

        //static Tensor Deserialize(BinaryReader reader)
        //{
        //    var t = new Tensor(Shape.Deserialize(reader));
        //    int valuesCount = reader.ReadInt32();
        //    t.Values = new float[valuesCount];
        //    for (int i = 0; i < valuesCount; ++i)
        //        t.Values[i] = reader.ReadSingle();
        //    return t;
        //}

        float GetFlat(int i);
        float& Get(int w, int h = 0, int d = 0, int n = 0);
        float& operator()(int w, int h = 0, int d = 0, int n = 0);
        float TryGet(float def, int w, int h = 0, int d = 0, int n = 0);
        void SetFlat(float value, int i);
        void Set(float value, int w, int h = 0, int d = 0, int n = 0);
        void TrySet(float value, int w, int h = 0, int d = 0, int n = 0);
        void CopyTo(Tensor& result, float tau = -1) const;
        void CopyBatchTo(int batchId, int targetBatchId, Tensor& result);
        void CopyDepthTo(int depthId, int batchId, int targetDepthId, int targetBatchId, Tensor& result);
        Tensor GetBatch(int batchId);
        Tensor GetDepth(int depthId, int batchId = 0);
        bool Equals(const Tensor& other, float epsilon = 0.00001f);
        float GetMaxData(int batch, int& maxIndex);
        void Elu(float alpha, Tensor& result);
        static void EluGradient(Tensor output, Tensor outputGradient, float alpha, Tensor& result);
        void Softmax(Tensor& result);
        static void SoftmaxGradient(Tensor output, Tensor outputGradient, Tensor& result);
		const Shape& GetShape() const { return m_Shape; }

        struct GPUData
        {
            /*CudaDeviceVariable<float> DeviceVar;
            CudaDeviceVariable<byte> ConvWorkspace;
            CudaDeviceVariable<byte> ConvBackWorkspace;
            CudaDeviceVariable<byte> ConvBackKernelWorkspace;*/

            ~GPUData()
            {
				/*DeviceVar ? .Dispose(); DeviceVar = null;
				ConvWorkspace ? .Dispose(); ConvWorkspace = null;
				ConvBackWorkspace ? .Dispose(); ConvBackWorkspace = null;
				ConvBackKernelWorkspace ? .Dispose(); ConvBackKernelWorkspace = null;*/
            }
		};

        void CopyToDevice() const;
        void CopyToHost() const;

	private:
        GPUData GpuData;
		TensorOpCpu* Op;
        mutable ELocation CurrentLocation;
        vector<float> Values;
		Shape m_Shape;

		static TensorOpCpu* g_OpCpu;
		static Random g_Rng;
	};
}
