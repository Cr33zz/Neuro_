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
        Tensor(istream& stream);
        Tensor(const Tensor& t);
        Tensor& operator=(const Tensor& t);

		static void SetDefaultOpMode(EOpMode mode);
        static void SetForcedOpMode(EOpMode mode);
        static void ClearForcedOpMode();

        void SetOpMode(EOpMode mode);

        uint32_t Width() const { return m_Shape.Width(); };
        uint32_t Height() const { return m_Shape.Height(); }
        uint32_t Depth() const { return m_Shape.Depth(); }
        uint32_t Batch() const { return m_Shape.Batch(); }
        uint32_t BatchLength() const { return m_Shape.Dim0Dim1Dim2; }
        uint32_t Length() const { return (uint32_t)m_Values.size(); }

        const string& Name() const { return m_Name; }

        vector<float>& GetValues();
        const vector<float>& GetValues() const;

        void SaveAsImage(const string& imageFile, bool denormalize) const;

        Tensor& FillWithRand(int seed = -1, float min = -1, float max = 1, uint32_t offset = 0);
        Tensor& FillWithRange(float start = 0, float increment = 1, uint32_t offset = 0);
        Tensor& FillWithValue(float value, uint32_t offset = 0);
        Tensor& FillWithFunc(const function<float()>& func, uint32_t offset = 0);

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
        void Sum(EAxis axis, int batch, Tensor& output) const;
		Tensor Avg(EAxis axis, int batch = -1) const;
        Tensor Max(EAxis axis, int batch = -1, Tensor* maxIndex = nullptr) const;
        Tensor Min(EAxis axis, int batch = -1, Tensor* minIndex = nullptr) const;
        // For Feature axis it will be batch index, for Sample axis it will be flat element index within a batch
        Tensor ArgMax(EAxis axis, int batch = -1) const;
        Tensor ArgMin(EAxis axis, int batch = -1) const;
        
        static Tensor MergeIntoBatch(const vector<Tensor>& tensors);
        // In case number of tensors is smaller than forcedDepth, first tensor will be repeated to account for missing tensors
        static Tensor MergeIntoDepth(const vector<Tensor>& tensors, uint32_t forcedDepth = 0);

		static void Concat(EAxis axis, const tensor_ptr_vec_t& inputs, Tensor& result);
        // This is reverse Concat operation
        void Split(EAxis axis, vector<Tensor>& outputs) const;

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
        Tensor Reshaped(const Shape& shape) const;
        void Reshaped(const Shape& shape, Tensor& output) const;
        void Reshape(const Shape& shape);

        // Changes shape and resizes values if neccessary.
        void Resize(const Shape& shape);

        // Create new tensor with different batch length and use current tensors values to fill the new tensor.
        // Number of batches will be the same as in source tensor.
        Tensor Resized(uint32_t width, uint32_t height = 1, uint32_t depth = 1) const;
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

        void Conv2D(const Tensor& kernels, uint32_t stride, uint32_t padding, Tensor& result) const;
        Tensor Conv2D(const Tensor& kernels, uint32_t stride, uint32_t padding) const;
        void Conv2DInputsGradient(const Tensor& gradient, const Tensor& kernels, uint32_t stride, uint32_t padding, Tensor& inputsGradient) const;
        void Conv2DKernelsGradient(const Tensor& input, const Tensor& gradient, uint32_t stride, uint32_t padding, Tensor& kernelsGradient) const;

        void Conv2DTransposed(const Tensor& kernels, uint32_t stride, uint32_t padding, Tensor& result) const;
        Tensor Conv2DTransposed(const Tensor& kernels, uint32_t outputDepth, uint32_t stride, uint32_t padding) const;
        void Conv2DTransposedInputsGradient(const Tensor& gradient, const Tensor& kernels, uint32_t stride, uint32_t padding, Tensor& inputsGradient) const;
        void Conv2DTransposedKernelsGradient(const Tensor& input, const Tensor& gradient, uint32_t stride, uint32_t padding, Tensor& kernelsGradient) const;

        void Pool2D(uint32_t filterSize, uint32_t stride, EPoolingMode type, uint32_t padding, Tensor& output) const;
        Tensor Pool2D(uint32_t filterSize, uint32_t stride, EPoolingMode type, uint32_t padding) const;
        void Pool2DGradient(const Tensor& output, const Tensor& input, const Tensor& outputGradient, uint32_t filterSize, uint32_t stride, EPoolingMode type, uint32_t padding, Tensor& result) const;

        void UpSample2D(uint32_t scaleFactor, Tensor& output) const;
        Tensor UpSample2D(uint32_t scaleFactor) const;
        void UpSample2DGradient(const Tensor& outputGradient, uint32_t scaleFactor, Tensor& inputGradient) const;

        void BatchNormalization(const Tensor& gamma, const Tensor& beta, const Tensor& runningMean, const Tensor& runningVar, Tensor& result) const;
        void BatchNormalizationTrain(const Tensor& gamma, const Tensor& beta, float momentum, Tensor& runningMean, Tensor& runningVar, Tensor& saveMean, Tensor& saveInvVariance, Tensor& result) const;
        void BatchNormalizationGradient(const Tensor& input, const Tensor& gamma, const Tensor& outputGradient, const Tensor& savedMean, const Tensor& savedInvVariance, Tensor& gammaGradient, Tensor& betaGradient, bool trainable, Tensor& inputGradient) const;

        void Dropout(float prob, Tensor& saveMask, Tensor& output) const;
        void DropoutGradient(const Tensor& outputGradient, const Tensor& savedMask, Tensor& inputGradient) const;

        string ToString() const;
        bool SameDimensionsExceptBatches(const Tensor& t) const;

        static pair<uint32_t, uint32_t> GetPadding(EPaddingMode paddingMode, uint32_t kernelWidth, uint32_t kernelHeight);
        static uint32_t GetPadding(EPaddingMode paddingMode, uint32_t kernelSize);
        static Shape GetPooling2DOutputShape(const Shape& inputShape, uint32_t kernelWidth, uint32_t kernelHeight, uint32_t stride, uint32_t paddingX, uint32_t paddingY);
        static Shape GetConvOutputShape(const Shape& inputShape, uint32_t kernelsNum, uint32_t kernelWidth, uint32_t kernelHeight, uint32_t stride, uint32_t paddingX, uint32_t paddingY);
        static Shape GetConvTransposeOutputShape(const Shape& inputShape, uint32_t outputDepth, uint32_t kernelWidth, uint32_t kernelHeight, uint32_t stride, uint32_t paddingX, uint32_t paddingY);

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

        void SaveBin(ostream& stream) const;
        void LoadBin(istream& stream);

        float& operator()(uint32_t w, uint32_t h = 0, uint32_t d = 0, uint32_t n = 0);
        float operator()(uint32_t w, uint32_t h = 0, uint32_t d = 0, uint32_t n = 0) const;

        float GetFlat(uint32_t i) const;
        float& Get(uint32_t w, uint32_t h = 0, uint32_t d = 0, uint32_t n = 0);
        float Get(uint32_t w, uint32_t h = 0, uint32_t d = 0, uint32_t n = 0) const;
        float TryGet(float def, int w, int h = 0, int d = 0, int n = 0) const;

        void SetFlat(float value, uint32_t i);
        void Set(float value, uint32_t w, uint32_t h = 0, uint32_t d = 0, uint32_t n = 0);
        void TrySet(float value, int w, int h = 0, int d = 0, int n = 0);

        void CopyTo(Tensor& result, float tau = 0) const;
        void CopyBatchTo(uint32_t batchId, uint32_t targetBatchId, Tensor& result) const;
        void CopyDepthTo(uint32_t depthId, uint32_t batchId, uint32_t targetDepthId, uint32_t targetBatchId, Tensor& result) const;
        Tensor GetBatch(uint32_t batchId) const;
        Tensor GetBatches(vector<uint32_t> batchIds) const;
        Tensor GetRandomBatches(uint32_t batchSize) const;
        void GetBatches(vector<uint32_t> batchIds, Tensor& result) const;
        Tensor GetDepth(uint32_t depthId, uint32_t batchId = 0) const;
        bool Equals(const Tensor& other, float epsilon = 0.00001f) const;        
        
        void Sigmoid(Tensor& result) const;
        void SigmoidGradient(const Tensor& output, const Tensor& outputGradient, Tensor& result) const;
        void Tanh(Tensor& result) const;
        void TanhGradient(const Tensor& output, const Tensor& outputGradient, Tensor& result) const;
        void ReLU(Tensor& result) const;
        void ReLUGradient(const Tensor& output, const Tensor& outputGradient, Tensor& result) const;
        void Elu(float alpha, Tensor& result) const;
        void EluGradient(const Tensor& output, const Tensor& outputGradient, float alpha, Tensor& result) const;
        void LeakyReLU(float alpha, Tensor& result) const;
        void LeakyReLUGradient(const Tensor& output, const Tensor& outputGradient, float alpha, Tensor& result) const;

        void Softmax(Tensor& result) const;
        void SoftmaxGradient(const Tensor& output, const Tensor& outputGradient, Tensor& result) const;

		const Shape& GetShape() const { return m_Shape; }

        void CopyToDevice() const;
        void CopyToHost() const;
        void OverrideHost() const;
        bool IsOnHost() const { return m_CurrentLocation == ELocation::Host; }
        bool IsOnDevice() const { return m_CurrentLocation == ELocation::Device; }
        
        const CudaDeviceVariable<float>& GetDeviceVar() const;
        const float* GetDevicePtr() const;
        float* GetDevicePtr();

        static TensorOpCpu* DefaultOp();
        static TensorOpCpu* ActiveOp();

        void DebugDumpValues(const string& outFile) const;
        void DebugRecoverValues(const string& inFile);

	private:
        struct GPUData
        {
            GPUData() {}
            ~GPUData();
            void Release();

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
		Shape m_Shape;
        mutable vector<float> m_Values;
        string m_Name;

        TensorOpCpu* Op() const { return g_ForcedOp ? g_ForcedOp : m_Op; }

		static TensorOpCpu* GetOpFromMode(EOpMode mode);

		static TensorOpCpu* g_DefaultOp;
        static TensorOpCpu* g_ForcedOp;
		static TensorOpCpu* g_OpCpu;
        static TensorOpCpu* g_OpMultiCpu;
        static TensorOpCpu* g_OpGpu;

        friend class TensorOpGpu;
	};
}
