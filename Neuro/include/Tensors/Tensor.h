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

    // This is HCHW tensor (meaning height is changing the most frequently and batch is changing the least frequently). This is just an interpretation of data.
    // Nothing prevents user from loading an image which is usually in NHWC (channels are changing the most frequently because images are usually stored as sequence
    // of RGB values for each pixel); however, before using it as an input to convolutional neural network such tensor should be converted to HCHW.
    class Tensor
    {
	public:
        Tensor(const string& name = "");
        explicit Tensor(const Shape& shape, const string& name = "");
        Tensor(const vector<float>&, const Shape& shape, const string& name = "");
        Tensor(const vector<float>&, const string& name = "");
        Tensor(const string& imageFile, bool normalize, bool grayScale = false, const string& name = "");
        Tensor(const Tensor& t);
        Tensor& operator=(const Tensor& t);

		static void SetDefaultOpMode(EOpMode mode);
        static void SetForcedOpMode(EOpMode mode);
        static void ClearForcedOpMode();

        void SetOpMode(EOpMode mode);

        uint Width() const { return m_Shape.Width(); };
        uint Height() const { return m_Shape.Height(); }
        uint Depth() const { return m_Shape.Depth(); }
        uint Batch() const { return m_Shape.Batch(); }
        uint BatchLength() const { return m_Shape.Dim0Dim1Dim2; }
        uint Length() const { return (uint)m_Values.size(); }

        const string& Name() const { return m_Name; }

        vector<float>& GetValues();
        const vector<float>& GetValues() const;

        void SaveAsImage(const string& imageFile, bool denormalize) const;

        Tensor& FillWithRand(int seed = -1, float min = -1, float max = 1);
        Tensor& FillWithRange(float start = 0, float increment = 1);
        Tensor& FillWithValue(float value);
        Tensor& FillWithFunc(const function<float()>& func);

        void Zero();

        // Converts to NCHW assuming the data is in NHWC format. Shape remains unchanged.
        Tensor ToNCHW() const;
        // Converts to NHWC assuming the data is in NCHW format. Shape remains unchanged.
        Tensor ToNHWC() const;
	
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

        Tensor Sum(EAxis axis, int batch = -1) const;
		Tensor Avg(EAxis axis, int batch = -1) const;
        Tensor Max(EAxis axis, int batch = -1, Tensor* maxIndex = nullptr) const;
        Tensor Min(EAxis axis, int batch = -1, Tensor* minIndex = nullptr) const;
        // For Feature axis it will be batch index, for Sample axis it will be flat element index within a batch
        Tensor ArgMax(EAxis axis, int batch = -1) const;
        Tensor ArgMin(EAxis axis, int batch = -1) const;
        
        static Tensor MergeIntoBatch(const vector<Tensor>& tensors);
        // In case number of tensors is smaller than forcedDepth, first tensor will be repeated to account for missing tensors
        static Tensor MergeIntoDepth(const vector<Tensor>& tensors, uint forcedDepth = 0);

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
        
        Tensor Transposed() const;
        void Transpose(Tensor& result) const;

        // Generates a new tensor with given dimensions and populate it with this tensor's values in index order.
        // One of dimensions can be -1, in that case it will be calculated based on remaining dimensions.
        Tensor Reshaped(Shape shape) const;
        void Reshape(Shape shape);

        // Create new tensor with different batch length and use current tensors values to fill the new tensor.
        // Number of batches will be the same as in source tensor.
        Tensor Resized(uint width, uint height = 1, uint depth = 1) const;
        Tensor FlattenHoriz() const;
        Tensor FlattenVert() const;
        void Rotated180(Tensor& result) const;
        Tensor Rotated180() const;

        // Returns norm value used to normalize data
        Tensor Normalized(EAxis axis, Tensor& result, ENormMode normMode = ENormMode::L2, Tensor* savedNorm = nullptr) const;
        // Returns normalized tensor
        Tensor Normalized(EAxis axis, ENormMode normMode = ENormMode::L2, Tensor* savedNorm = nullptr) const;
        pair<Tensor, Tensor> NormalizedMinMax(EAxis axis, Tensor& result, float scaleMin = 0, float scaleMax = 1, Tensor* savedMin = nullptr, Tensor* savedMax = nullptr) const;
        Tensor NormalizedMinMax(EAxis axis, float scaleMin = 0, float scaleMax = 1, Tensor* savedMin = nullptr, Tensor* savedMax = nullptr) const;
        pair<Tensor, Tensor> Standardized(EAxis axis, Tensor& result, Tensor* mean = nullptr, Tensor* invVariance = nullptr) const;
        Tensor Standardized(EAxis axis, Tensor* mean = nullptr, Tensor* invVariance = nullptr) const;

        void Conv2D(const Tensor& kernels, uint stride, uint padding, Tensor& result) const;
        Tensor Conv2D(const Tensor& kernels, uint stride, uint padding) const;
        void Conv2DInputsGradient(const Tensor& gradient, const Tensor& kernels, uint stride, uint padding, Tensor& inputsGradient) const;
        void Conv2DKernelsGradient(const Tensor& input, const Tensor& gradient, uint stride, uint padding, Tensor& kernelsGradient) const;

        void Conv2DTransposed(const Tensor& kernels, uint stride, uint padding, Tensor& result) const;
        Tensor Conv2DTransposed(const Tensor& kernels, uint outputDepth, uint stride, uint padding) const;
        void Conv2DTransposedInputsGradient(const Tensor& gradient, const Tensor& kernels, uint stride, uint padding, Tensor& inputsGradient) const;
        void Conv2DTransposedKernelsGradient(const Tensor& input, const Tensor& gradient, uint stride, uint padding, Tensor& kernelsGradient) const;

        void Pool2D(uint filterSize, uint stride, EPoolingMode type, uint padding, Tensor& output) const;
        Tensor Pool2D(uint filterSize, uint stride, EPoolingMode type, uint padding) const;
        void Pool2DGradient(const Tensor& output, const Tensor& input, const Tensor& outputGradient, uint filterSize, uint stride, EPoolingMode type, uint padding, Tensor& result) const;

        void UpSample2D(uint scaleFactor, Tensor& output) const;
        Tensor UpSample2D(uint scaleFactor) const;
        void UpSample2DGradient(const Tensor& outputGradient, uint scaleFactor, Tensor& inputGradient) const;

        void BatchNormalization(const Tensor& gamma, const Tensor& beta, const Tensor& runningMean, const Tensor& runningVar, Tensor& result) const;
        void BatchNormalizationTrain(const Tensor& gamma, const Tensor& beta, float momentum, Tensor& runningMean, Tensor& runningVar, Tensor& saveMean, Tensor& saveInvVariance, Tensor& result) const;
        void BatchNormalizationGradient(const Tensor& input, const Tensor& gamma, const Tensor& outputGradient, const Tensor& savedMean, const Tensor& savedInvVariance, Tensor& gammaGradient, Tensor& betaGradient, Tensor& inputGradient) const;

        void Dropout(float prob, Tensor& saveMask, Tensor& output) const;
        void DropoutGradient(const Tensor& outputGradient, const Tensor& savedMask, Tensor& inputGradient) const;

        string ToString() const;
        bool SameDimensionsExceptBatches(const Tensor& t) const;

        static pair<uint, uint> GetPadding(EPaddingMode paddingMode, uint kernelWidth, uint kernelHeight);
        static uint GetPadding(EPaddingMode paddingMode, uint kernelSize);
        static Shape GetPooling2DOutputShape(const Shape& inputShape, uint kernelWidth, uint kernelHeight, uint stride, uint paddingX, uint paddingY);
        static Shape GetConvOutputShape(const Shape& inputShape, uint kernelsNum, uint kernelWidth, uint kernelHeight, uint stride, uint paddingX, uint paddingY);
        static Shape GetConvTransposeOutputShape(const Shape& inputShape, uint outputDepth, uint kernelWidth, uint kernelHeight, uint stride, uint paddingX, uint paddingY);

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
        //    for (uint i = 0; i < valuesCount; ++i)
        //        t.Values[i] = reader.ReadSingle();
        //    return t;
        //}

        float& operator()(uint w, uint h = 0, uint d = 0, uint n = 0);
        float operator()(uint w, uint h = 0, uint d = 0, uint n = 0) const;

        float GetFlat(uint i) const;
        float& Get(uint w, uint h = 0, uint d = 0, uint n = 0);
        float Get(uint w, uint h = 0, uint d = 0, uint n = 0) const;
        float TryGet(float def, int w, int h = 0, int d = 0, int n = 0) const;

        void SetFlat(float value, uint i);
        void Set(float value, uint w, uint h = 0, uint d = 0, uint n = 0);
        void TrySet(float value, int w, int h = 0, int d = 0, int n = 0);

        void CopyTo(Tensor& result, float tau = 0) const;
        void CopyBatchTo(int batchId, int targetBatchId, Tensor& result) const;
        void CopyDepthTo(int depthId, int batchId, int targetDepthId, int targetBatchId, Tensor& result) const;
        Tensor GetBatch(int batchId) const;
        Tensor GetDepth(int depthId, int batchId = 0) const;
        bool Equals(const Tensor& other, float epsilon = 0.00001f) const;        
        void Elu(float alpha, Tensor& result) const;
        void EluGradient(const Tensor& output, const Tensor& outputGradient, float alpha, Tensor& result) const;
        void Softmax(Tensor& result) const;
        void SoftmaxGradient(const Tensor& output, const Tensor& outputGradient, Tensor& result) const;
		const Shape& GetShape() const { return m_Shape; }

        void CopyToDevice() const;
        void CopyToHost() const;
        void OverrideHost() const;
        
        const CudaDeviceVariable<float>& GetDeviceVar() const;
        const float* GetDevicePtr() const;
        float* GetDevicePtr();

	private:
        struct GPUData
        {
            GPUData() {}
            ~GPUData();

            void UpdateWorkspace(CudaDeviceVariable<char>*& workspace, size_t size);

            CudaDeviceVariable<float>* m_DeviceVar = nullptr;
            CudaDeviceVariable<char>* m_ConvWorkspace = nullptr;
            CudaDeviceVariable<char>* m_ConvBackWorkspace = nullptr;
            CudaDeviceVariable<char>* m_ConvBackKernelWorkspace = nullptr;

        private:
            GPUData(const GPUData&);
            GPUData& operator=(const GPUData&);
        };

        mutable GPUData m_GpuData;
		TensorOpCpu* m_Op;
        mutable ELocation m_CurrentLocation;
        mutable vector<float_t> m_Values;
		Shape m_Shape;
        string m_Name;

        TensorOpCpu* Op() const { return g_ForcedOp ? g_ForcedOp : m_Op; }

		static TensorOpCpu* GetOpFromMode(EOpMode mode);

		static TensorOpCpu* g_DefaultOpCpu;
        static TensorOpCpu* g_ForcedOp;
		static TensorOpCpu* g_OpCpu;
        static TensorOpCpu* g_OpMultiCpu;
        static TensorOpCpu* g_OpGpu;

        static void ImageLibInit();
        static bool g_ImageLibInitialized;

        friend class TensorOpGpu;
	};
}
