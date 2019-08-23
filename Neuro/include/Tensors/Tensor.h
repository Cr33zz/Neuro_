#pragma once

#include <functional>
#include <limits>
#include <string>
#include <sstream>
#include <vector>
#include "assert.h"

#include "Types.h"
#include "Tensors/Shape.h"

namespace Neuro
{
	class TensorOpCpu;
	class Random;
    template<typename T> class CudaDeviceVariable;

	using namespace std;

    class Tensor
    {
	public:
		Tensor();
        explicit Tensor(const Shape& shape);
        Tensor(const vector<float>&, const Shape& shape);
        Tensor(const vector<float>&);
        Tensor(const Tensor& t);
        Tensor(string bmpFile, bool grayScale);

		static void SetDefaultOpMode(EOpMode mode);
        void SetOpMode(EOpMode mode);

        int Width() const { return m_Shape.Width(); };
		int Height() const { return m_Shape.Height(); }
		int Depth() const { return m_Shape.Depth(); }
		int Batch() const { return m_Shape.Batch(); }
		int BatchLength() const { return m_Shape.Dim0Dim1Dim2; }
		int Length() const { return (int)m_Values.size(); }

        vector<float>& GetValues();
        const vector<float>& GetValues() const;

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
        Tensor& FillWithFunc(const function<float()>& func);

        void Zero();
	
	private:
        void Mul(bool transposeT, const Tensor& t, Tensor& result) const;
        Tensor Mul(bool transposeT, const Tensor& t) const;

	public:
        void Mul(const Tensor& t, Tensor& result) const;
        Tensor Mul(const Tensor& t) const;
        void MulElem(const Tensor& t, Tensor& result) const;
        Tensor MulElem(const Tensor& t) const;
        void Mul(float v, Tensor& result) const;
        Tensor Mul(float v) const;
        void Div(const Tensor& t, Tensor& result) const;
        Tensor Div(const Tensor& t) const;
        void Div(float v, Tensor& result) const;
        Tensor Div(float v) const;
        void Add(float alpha, float beta, const Tensor& t, Tensor& result) const;
        void Add(const Tensor& t, Tensor& result) const;
        Tensor Add(const Tensor& t) const;
        Tensor Add(float alpha, float beta, const Tensor& t) const;
        void Add(float v, Tensor& result) const;
        Tensor Add(float v) const;
        void Sub(const Tensor& t, Tensor& result) const;
        Tensor Sub(const Tensor& t) const;
        void Sub(float v, Tensor& result) const;
        Tensor Sub(float v) const;
        void Negated(Tensor& result) const;
        Tensor Negated() const;
        void Clipped(float min, float max, Tensor& result) const;
        Tensor Clipped(float min, float max) const;
        Tensor DiagFlat() const;

        void Map(const function<float(float)>& func, Tensor& result) const;
		Tensor Map(const function<float(float)>& func) const;
		void Map(const function<float(float, float)>&, const Tensor& other, Tensor& result) const;
		Tensor Map(const function<float(float, float)>& func, const Tensor& other) const;

        Tensor SumBatches() const;
        float Sum(int batch = -1) const;
        Tensor SumPerBatch() const;

        Tensor AvgBatches() const;
		float Avg(int batch = -1) const;
        Tensor AvgPerBatch() const;

		float Max(int batch = -1) const;
        Tensor MaxPerBatch() const;

        static Tensor MergeIntoBatch(const vector<Tensor>& tensors);
        // In case number of tensors is smaller than forcedDepth, first tensor will be repeated to account for missing tensors
        static Tensor MergeIntoDepth(const vector<Tensor>& tensors, int forcedDepth = 0);

		// This operation will concatenate elements of all input tensors separately for each batch
		static void Concat(const tensor_ptr_vec_t& inputs, Tensor& result);
        // This is reverse Concat operation
        void Split(vector<Tensor>& outputs) const;

        static void MergeMin(const tensor_ptr_vec_t& inputs, Tensor& result);
        static void MergeMax(const tensor_ptr_vec_t& inputs, Tensor& result);
        static void MergeSum(const tensor_ptr_vec_t& inputs, Tensor& result);
        static void MergeAvg(const tensor_ptr_vec_t& inputs, Tensor& result);
        static void MergeMinMaxGradient(const Tensor& output, const tensor_ptr_vec_t& inputs, const Tensor& outputGradient, vector<Tensor>& results);
        static void MergeSumGradient(const Tensor& output, const tensor_ptr_vec_t& inputs, const Tensor& outputGradient, vector<Tensor>& results);
        static void MergeAvgGradient(const Tensor& output, const tensor_ptr_vec_t& inputs, const Tensor& outputGradient, vector<Tensor>& results);

        void Normalized(Tensor& result) const;
        // ArgMax will return local index inside given batch if batch is not -1
        int ArgMax(int batch = -1) const;
        Tensor ArgMaxPerBatch() const;
        Tensor Transposed() const;
        void Transpose(Tensor& result) const;

        // Generates a new tensor with given dimensions and populate it with this tensor's values in index order.
        // One of dimensions can be -1, in that case it will be calculated based on remaining dimensions.
        Tensor Reshaped(Shape shape) const;
        void Reshape(Shape shape);

        // Create new tensor with different batch length and use current tensors values to fill the new tensor.
        // Number of batches will be the same as in source tensor.
        Tensor Resized(int width, int height = 1, int depth = 1) const;
        Tensor FlattenHoriz() const;
        Tensor FlattenVert() const;
        void Rotated180(Tensor& result) const;
        Tensor Rotated180() const;

        void Conv2D(const Tensor& kernels, int stride, int padding, Tensor& result) const;
        Tensor Conv2D(const Tensor& kernels, int stride, int padding) const;
        void Conv2DInputsGradient(const Tensor& gradient, const Tensor& kernels, int stride, int padding, Tensor& inputsGradient) const;
        void Conv2DKernelsGradient(const Tensor& input, const Tensor& gradient, int stride, int padding, Tensor& kernelsGradient) const;

        void Conv2DTransposed(const Tensor& kernels, int stride, int padding, Tensor& result) const;
        Tensor Conv2DTransposed(const Tensor& kernels, int outputDepth, int stride, int padding) const;
        void Conv2DTransposedInputsGradient(const Tensor& gradient, const Tensor& kernels, int stride, int padding, Tensor& inputsGradient) const;
        void Conv2DTransposedKernelsGradient(const Tensor& input, const Tensor& gradient, int stride, int padding, Tensor& kernelsGradient) const;

        void Pool2D(int filterSize, int stride, EPoolingMode type, int padding, Tensor& result) const;
        Tensor Pool2D(int filterSize, int stride, EPoolingMode type, int padding) const;
        // Assuming result matrix is of the dimensions of input to pooling layer
        void Pool2DGradient(const Tensor& output, const Tensor& input, const Tensor& outputGradient, int filterSize, int stride, EPoolingMode type, int padding, Tensor& result) const;

        string ToString() const;
        bool SameDimensionsExceptBatches(const Tensor& t) const;

        static pair<int,int> GetPadding(EPaddingMode paddingMode, int kernelWidth, int kernelHeight);
        static int GetPadding(EPaddingMode paddingMode, int kernelSize);
        //static void GetPaddingParams(EPaddingMode type, int width, int height, int kernelWidth, int kernelHeight, int stride, int& outHeight, int& outWidth, int& paddingX, int& paddingY);
        static Shape GetPooling2DOutputShape(const Shape& inputShape, int kernelWidth, int kernelHeight, int stride, int paddingX, int paddingY);
        static Shape GetConvOutputShape(const Shape& inputShape, int kernelsNum, int kernelWidth, int kernelHeight, int stride, int paddingX, int paddingY);
        static Shape GetConvTransposeOutputShape(const Shape& inputShape, int outputDepth, int kernelWidth, int kernelHeight, int stride, int paddingX, int paddingY);

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

        float& operator()(int w, int h = 0, int d = 0, int n = 0);
        float operator()(int w, int h = 0, int d = 0, int n = 0) const;

        float GetFlat(int i);
        float& Get(int w, int h = 0, int d = 0, int n = 0);
        float Get(int w, int h = 0, int d = 0, int n = 0) const;
        float TryGet(float def, int w, int h = 0, int d = 0, int n = 0) const;

        void SetFlat(float value, int i);
        void Set(float value, int w, int h = 0, int d = 0, int n = 0);
        void TrySet(float value, int w, int h = 0, int d = 0, int n = 0);

        void CopyTo(Tensor& result, float tau = 0) const;
        void CopyBatchTo(int batchId, int targetBatchId, Tensor& result) const;
        void CopyDepthTo(int depthId, int batchId, int targetDepthId, int targetBatchId, Tensor& result) const;
        Tensor GetBatch(int batchId) const;
        Tensor GetDepth(int depthId, int batchId = 0) const;
        bool Equals(const Tensor& other, float epsilon = 0.00001f) const;
        float GetMaxData(int batch, int& maxIndex) const;
        void Elu(float alpha, Tensor& result) const;
        void EluGradient(const Tensor& output, const Tensor& outputGradient, float alpha, Tensor& result) const;
        void Softmax(Tensor& result) const;
        void SoftmaxGradient(const Tensor& output, const Tensor& outputGradient, Tensor& result) const;
		const Shape& GetShape() const { return m_Shape; }

        struct GPUData
        {
            ~GPUData();

            void UpdateWorkspace(CudaDeviceVariable<char>*& workspace, size_t size);            

            CudaDeviceVariable<float>* m_DeviceVar = nullptr;
            CudaDeviceVariable<char>* m_ConvWorkspace = nullptr;
            CudaDeviceVariable<char>* m_ConvBackWorkspace = nullptr;
            CudaDeviceVariable<char>* m_ConvBackKernelWorkspace = nullptr;
		};

        void CopyToDevice() const;
        void CopyToHost() const;
        void OverrideHost() const;
        
        const CudaDeviceVariable<float>& GetDeviceVar() const;
        const float* GetDevicePtr() const;
        float* GetDevicePtr();

	private:
        mutable GPUData m_GpuData;
		TensorOpCpu* m_Op;
        mutable ELocation m_CurrentLocation;
        mutable vector<float_t> m_Values;
		Shape m_Shape;

		static TensorOpCpu* GetOpMode(EOpMode mode);

		static TensorOpCpu* g_DefaultOpCpu;
		static TensorOpCpu* g_OpCpu;
        static TensorOpCpu* g_OpMultiCpu;
        static TensorOpCpu* g_OpGpu;

        friend class TensorOpGpu;
	};
}
