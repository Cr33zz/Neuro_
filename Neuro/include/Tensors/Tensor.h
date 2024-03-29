﻿#pragma once

#include <functional>
#include <limits>
#include <string>
#include <sstream>
#include <vector>
#include "assert.h"

#include "Types.h"
#include "Tensors/Shape.h"
#include "Tensors/Storage.h"

#pragma warning(push)
#pragma warning(disable:4251)

namespace Neuro
{
	class TensorOpCpu;
	class Random;
    template<typename T> class CudaDeviceVariable;

	using namespace std;

    // This is HCHW tensor (meaning height is changing the most frequently and batch is changing the least frequently). This is just an interpretation of data.
    // Nothing prevents user from loading an image which is usually in NHWC (channels are changing the most frequently because images are usually stored as sequence
    // of RGB values for each pixel); however, before using it as an input to convolutional neural network such tensor should be converted to HCHW.
    class NEURO_DLL_EXPORT Tensor
    {
	public:
        explicit Tensor(const Shape& shape = Shape(0), const string& name = "", EStorageType storageType = ST_Default);
        Tensor(const vector<float>&, const Shape& shape, const string& name = "", EStorageType storageType = ST_Default);
        Tensor(const vector<float>&, const string& name = "", EStorageType storageType = ST_Default);
        Tensor(istream& stream, EStorageType storageType = ST_Default);
        Tensor(const Tensor& t);
        Tensor(Tensor&& t);
        Tensor& operator=(const Tensor& t);
        Tensor& operator=(Tensor&& t);

		static void SetDefaultOpMode(EOpMode mode);
        static void SetForcedOpMode(EOpMode mode);
        static void ClearForcedOpMode();

        void SetOpMode(EOpMode mode);

        uint32_t Width() const { return m_Shape.Width(); };
        uint32_t Height() const { return m_Shape.Height(); }
        uint32_t Depth() const { return m_Shape.Depth(); }
        uint32_t Batch() const { return m_Shape.Batch(); }
        uint32_t Len(size_t dim) const { return m_Shape.Dimensions[dim]; }
        uint32_t NDim() const { return m_Shape.NDim; }
        uint32_t Stride(size_t dim) const { return m_Shape.Stride[dim]; }
        uint32_t BatchLength() const { return m_Shape.Stride[3]; }
        uint32_t Length() const { return m_Shape.Length; }

        const string& Name() const { return m_Name; }
        void Name(const string& name);

        float* Values();
        const float* Values() const;
        void SetStorageType(int type);

        bool Validate() const;

        void SaveAsImage(const string& imageFile, bool denormalize, uint32_t maxCols = 0) const;
        void SaveAsH5(const string& h5File) const;
        void LoadFromH5(const string& h5File);

        Tensor& FillWithRand(int seed = -1, float min = -1, float max = 1, uint32_t offset = 0);
        Tensor& FillWithRange(float start = 0, float increment = 1, uint32_t offset = 0);
        Tensor& FillWithValue(float value, uint32_t offset = 0);
        Tensor& FillWithFunc(const function<float()>& func, uint32_t offset = 0);

        void Zero();
        void One();

        // Converts to NCHW assuming the data is in NHWC format. Shape remains unchanged.
        Tensor ToNCHW() const;
        // Converts to NHWC assuming the data is in NCHW format. Shape remains unchanged.
        Tensor ToNHWC() const;
        Tensor ToGrayScale(uint32_t depth = 1) const;
        Tensor ToRGB() const;

        // RGB -> YUV conversion is only valid if pixel values are in [0,1]
        Tensor RGBToYUV() const;
        // YUV -> RGB conversion is only valid if pixel values are in [0,1] for Y and [-0.5,0.5] for U and V
        Tensor YUVToRGB() const;
	
	    void MatMul(bool transpose, const Tensor& t, bool transposeT, Tensor& result) const;
        Tensor MatMul(bool transpose, const Tensor& t, bool transposeT) const;

        // Performs A*A' (if transpose is false) or A'*A (otherwise)
        void MatMul(bool transpose, Tensor& result) const;
        Tensor MatMul(bool transpose) const;

        void MatMul(const Tensor& t, Tensor& result) const;
        Tensor MatMul(const Tensor& t) const;
        void MulElem(const Tensor& t, Tensor& result) const;
        Tensor MulElem(const Tensor& t) const;
        float Dot(const Tensor& t) const;
        void Mul(float v, Tensor& result) const;
        Tensor Mul(float v) const;
        void Scale(float v);
        void Div(const Tensor& t, Tensor& result) const;
        void Div(float alpha, float beta, const Tensor& t, Tensor& result) const;
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
        void Inversed(float alpha, Tensor& result) const;
        Tensor Inversed(float alpha = 1.f) const;
        void Clip(float min, float max, Tensor& result) const;
        Tensor Clip(float min, float max) const;
        void ClipGradient(const Tensor& input, float min, float max, const Tensor& outputGradient, Tensor& inputGradient) const;

        void ExtractSubTensor2D(uint32_t widthOffset, uint32_t heightOffset, Tensor& output, bool clampAllowed = false) const;
        void FuseSubTensor2D(uint32_t widthOffset, uint32_t heightOffset, Tensor& output, bool clampAllowed = false, bool add = false) const;

        Tensor DiagFlat() const;

        Tensor Pow(float power) const;
        void Pow(float power, Tensor& result) const;
        void PowGradient(const Tensor& input, float power, const Tensor& outputGradient, Tensor& inputGradient) const;

        Tensor Abs() const;
        void Abs(Tensor& result) const;
        void AbsGradient(const Tensor& input, const Tensor& outputGradient, Tensor& inputGradient) const;

        Tensor Sqrt() const;
        void Sqrt(Tensor& output) const;
        Tensor Log() const;
        void Log(Tensor& output) const;

        void Map(const function<float(float)>& func, Tensor& result) const;
		Tensor Map(const function<float(float)>& func) const;
		void Map(const function<float(float, float)>&, const Tensor& other, Tensor& result) const;
		Tensor Map(const function<float(float, float)>& func, const Tensor& other) const;

        Tensor AbsSum(EAxis axis) const;
        void AbsSum(EAxis axis, Tensor& output) const;
        Tensor Sum(EAxis axis) const;
        void Sum(EAxis axis, Tensor& output) const;
		Tensor Mean(EAxis axis) const;
        void Mean(EAxis axis, Tensor& output) const;
        Tensor Max(EAxis axis, Tensor* maxIndex = nullptr) const;
        Tensor Min(EAxis axis, Tensor* minIndex = nullptr) const;
        // For Feature axis it will be batch index, for Sample axis it will be flat element index within a batch
        Tensor ArgMax(EAxis axis) const;
        Tensor ArgMin(EAxis axis) const;
        
        static Tensor MergeIntoBatch(const vector<Tensor>& tensors);
        // In case number of tensors is smaller than forcedDepth, first tensor will be repeated to account for missing tensors
        static Tensor MergeIntoDepth(const vector<Tensor>& tensors, uint32_t forcedDepth = 0);

        static void Concat(EAxis axis, const const_tensor_ptr_vec_t& inputs, Tensor& output);
        // This is reverse Concat operation
        void Split(EAxis axis, tensor_ptr_vec_t& outputs) const;

        static void MergeMin(const const_tensor_ptr_vec_t& inputs, Tensor& output);
        static void MergeMax(const const_tensor_ptr_vec_t& inputs, Tensor& output);
        static void MergeSum(const const_tensor_ptr_vec_t& inputs, Tensor& output);
        static void MergeAvg(const const_tensor_ptr_vec_t& inputs, Tensor& output);
        static void MergeMinMaxGradient(const Tensor& output, const const_tensor_ptr_vec_t& inputs, const Tensor& outputGradient, tensor_ptr_vec_t& results);
        static void MergeSumGradient(const Tensor& output, const const_tensor_ptr_vec_t& inputs, const Tensor& outputGradient, tensor_ptr_vec_t& results);
        static void MergeAvgGradient(const Tensor& output, const const_tensor_ptr_vec_t& inputs, const Tensor& outputGradient, tensor_ptr_vec_t& results);

        static vector<EAxis> FillUpTranposeAxis(const vector<EAxis>& axes);
        
        // Axis specifies new order of axis (dimensions) using input tensor axis nomenclature
        Tensor Transpose(const vector<EAxis>& permutation) const;
        void Transpose(const vector<EAxis>& permutation, Tensor& output) const;

        Tensor Transpose() const;
        void Transpose(Tensor& output) const;

        // Generates a new tensor with given dimensions and populate it with this tensor's values in index order.
        // One of dimensions can be -1, in that case it will be calculated based on remaining dimensions.
        Tensor Reshaped(const Shape& shape) const;
        void Reshaped(const Shape& shape, Tensor& output) const;
        void Reshape(const Shape& shape);

        // Changes shape and resizes values if necessary.
        void Resize(const Shape& shape);
        void Resize(uint32_t length);
        void ResizeBatch(uint32_t batch);

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

        float L1Norm() const;
        float L2Norm() const;
        float SquaredL2Norm() const;

        void ConstantPad2D(uint32_t left, uint32_t right, uint32_t top, uint32_t bottom, float value, Tensor& output) const;
        Tensor ConstantPad2D(uint32_t left, uint32_t right, uint32_t top, uint32_t bottom, float value) const;
        void ReflectPad2D(uint32_t left, uint32_t right, uint32_t top, uint32_t bottom, Tensor& output) const;
        Tensor ReflectPad2D(uint32_t left, uint32_t right, uint32_t top, uint32_t bottom) const;
        void LinearRampPad2D(uint32_t left, uint32_t right, uint32_t top, uint32_t bottom, float endValue, Tensor& output) const;
        Tensor LinearRampPad2D(uint32_t left, uint32_t right, uint32_t top, uint32_t bottom, float endValue) const;
        void Pad2DGradient(const Tensor& gradient, uint32_t left, uint32_t right, uint32_t top, uint32_t bottom, Tensor& inputGradient) const;

        Tensor Roll2D(int xShift, int yShift) const;
        void Roll2D(int xShift, int yShift, Tensor& output) const;
        void Roll2DInPlace(int xShift, int yShift);

        void Conv2D(const Tensor& kernels, uint32_t stride, uint32_t padding, EDataFormat dataFormat, Tensor& output) const;
        Tensor Conv2D(const Tensor& kernels, uint32_t stride, uint32_t padding, EDataFormat dataFormat) const;
        void Conv2DBiasActivation(const Tensor& kernels, uint32_t stride, uint32_t padding, const Tensor& bias, EActivation activation, float activationAlpha, Tensor& output) const;
        Tensor Conv2DBiasActivation(const Tensor& kernels, uint32_t stride, uint32_t padding, const Tensor& bias, EActivation activation, float activationAlpha) const;
        void Conv2DBiasGradient(const Tensor& gradient, Tensor& biasGradient) const;
        void Conv2DInputsGradient(const Tensor& gradient, const Tensor& kernels, uint32_t stride, uint32_t padding, EDataFormat dataFormat, Tensor& inputsGradient) const;
        void Conv2DKernelsGradient(const Tensor& input, const Tensor& gradient, uint32_t stride, uint32_t padding, EDataFormat dataFormat, Tensor& kernelsGradient) const;

        void Conv2DTransposed(const Tensor& kernels, uint32_t stride, uint32_t padding, EDataFormat dataFormat, Tensor& result) const;
        Tensor Conv2DTransposed(const Tensor& kernels, uint32_t outputDepth, uint32_t stride, uint32_t padding, EDataFormat dataFormat) const;
        void Conv2DTransposedInputsGradient(const Tensor& gradient, const Tensor& kernels, uint32_t stride, uint32_t padding, EDataFormat dataFormat, Tensor& inputsGradient) const;
        void Conv2DTransposedKernelsGradient(const Tensor& input, const Tensor& gradient, uint32_t stride, uint32_t padding, EDataFormat dataFormat, Tensor& kernelsGradient) const;

        void Pool2D(uint32_t filterSize, uint32_t stride, EPoolingMode type, uint32_t padding, EDataFormat dataFormat, Tensor& output) const;
        Tensor Pool2D(uint32_t filterSize, uint32_t stride, EPoolingMode type, uint32_t padding, EDataFormat dataFormat) const;
        void Pool2DGradient(const Tensor& output, const Tensor& input, const Tensor& outputGradient, uint32_t filterSize, uint32_t stride, EPoolingMode type, uint32_t padding, EDataFormat dataFormat, Tensor& result) const;

        void UpSample2D(uint32_t scaleFactor, Tensor& output) const;
        Tensor UpSample2D(uint32_t scaleFactor) const;
        void UpSample2DGradient(const Tensor& outputGradient, uint32_t scaleFactor, Tensor& inputGradient) const;

        void BatchNorm(const Tensor& gamma, const Tensor& beta, float epsilon, const Tensor* runningMean, const Tensor* runningVar, Tensor& result) const;
        void BatchNormTrain(const Tensor& gamma, const Tensor& beta, float momentum, float epsilon, Tensor* runningMean, Tensor* runningVar, Tensor& saveMean, Tensor& saveInvVariance, Tensor& result) const;
        void BatchNormGradient(const Tensor& input, const Tensor& gamma, float epsilon, const Tensor& outputGradient, const Tensor& savedMean, const Tensor& savedInvVariance, Tensor& gammaGradient, Tensor& betaGradient, bool trainable, Tensor& inputGradient) const;

        void InstanceNorm(const Tensor& gamma, const Tensor& beta, float epsilon, Tensor& result) const;
        void InstanceNormTrain(const Tensor& gamma, const Tensor& beta, float epsilon, Tensor& saveMean, Tensor& saveInvVariance, Tensor& result) const;
        void InstanceNormGradient(const Tensor& input, const Tensor& gamma, float epsilon, const Tensor& outputGradient, const Tensor& savedMean, const Tensor& savedInvVariance, Tensor& gammaGradient, Tensor& betaGradient, bool trainable, Tensor& inputGradient) const;
        
        void Dropout(float prob, Tensor& saveMask, Tensor& output) const;
        void DropoutGradient(const Tensor& outputGradient, float prob, const Tensor& savedMask, Tensor& inputGradient) const;

        string ToString() const;
        bool SameDimensionsExceptBatches(const Tensor& t) const;
        bool SameDimensionsOrOne(const Tensor& t) const;

        static pair<uint32_t, uint32_t> GetPadding(EPaddingMode paddingMode, uint32_t kernelWidth, uint32_t kernelHeight);
        static uint32_t GetPadding(EPaddingMode paddingMode, uint32_t kernelSize);
        static Shape GetPooling2DOutputShape(const Shape& inputShape, uint32_t kernelWidth, uint32_t kernelHeight, uint32_t stride, uint32_t paddingX, uint32_t paddingY, EDataFormat dataFormat);
        static Shape GetConvOutputShape(const Shape& inputShape, uint32_t kernelsNum, uint32_t kernelWidth, uint32_t kernelHeight, uint32_t stride, uint32_t paddingX, uint32_t paddingY, EDataFormat dataFormat);
        static Shape GetConvTransposeOutputShape(const Shape& inputShape, uint32_t outputDepth, uint32_t kernelWidth, uint32_t kernelHeight, uint32_t stride, uint32_t paddingX, uint32_t paddingY, EDataFormat dataFormat);

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

        void CopyTo(Tensor& target, float tau = 0) const;
        void CopyTo(size_t offset, Tensor& target, size_t targetOffset, size_t elementsNum) const;
        void CopyBatchTo(uint32_t batchId, uint32_t targetBatchId, Tensor& target) const;
        void CopyDepthTo(uint32_t depthId, uint32_t batchId, uint32_t targetDepthId, uint32_t targetBatchId, Tensor& target) const;
        Tensor GetBatch(uint32_t batchId) const;
        Tensor GetBatches(vector<uint32_t> batchIds) const;
        Tensor GetRandomBatches(uint32_t batchSize) const;
        void GetBatches(vector<uint32_t> batchIds, Tensor& result) const;
        Tensor GetDepth(uint32_t depthId, uint32_t batchId = 0) const;
        bool Equals(const Tensor& other, float epsilon = 0.00001f) const;        
        
        void Activation(EActivation activation, float coeff, Tensor& output) const;
        void ActivationGradient(EActivation activation, float coeff, const Tensor& output, const Tensor& outputGradient, Tensor& inputGradient) const;
        void Sigmoid(Tensor& result) const;
        void SigmoidGradient(const Tensor& output, const Tensor& outputGradient, Tensor& inputGradient) const;
        void Tanh(Tensor& result) const;
        void TanhGradient(const Tensor& output, const Tensor& outputGradient, Tensor& inputGradient) const;
        void ReLU(Tensor& result) const;
        void ReLUGradient(const Tensor& output, const Tensor& outputGradient, Tensor& inputGradient) const;
        void Elu(float alpha, Tensor& result) const;
        void EluGradient(const Tensor& output, const Tensor& outputGradient, float alpha, Tensor& inputGradient) const;
        void LeakyReLU(float alpha, Tensor& result) const;
        void LeakyReLUGradient(const Tensor& output, const Tensor& outputGradient, float alpha, Tensor& inputGradient) const;
        void Softmax(Tensor& result) const;
        void SoftmaxGradient(const Tensor& output, const Tensor& outputGradient, Tensor& inputGradient) const;

		const Shape& GetShape() const { return m_Shape; }

        bool TryDeviceAllocate() const;
        bool TryDeviceRelease();
        //void ScheduleOffload() const;
        void Offload(bool force) const;
        void Prefetch() const;
        void ResetDeviceRef(size_t n = 0) const;
        void IncDeviceRef(size_t n = 1);
        void DecDeviceRef(size_t n = 1);
        void ResetRef(size_t n = 0) const;
        void IncRef(size_t n = 1);
        void DecRef(size_t n = 1);
        void ReleaseData();
        void CopyToDevice() const;
        void CopyToHost(bool allowAlloc = false) const;
        /// Sync will copy data from device to host but it won't change location (useful for read-only operations performed on CPU)
        void SyncToHost() const; 
        /// Use whatever data there is on the host (usually used for output tensors so copy can be avoided)
        void OverrideHost();
        /// Use whatever data there is on the device (usually used for output tensors so copy can be avoided)
        void OverrideDevice();
        bool IsOnHost() const { return m_Storage.Location() == Host; }
        bool IsOnDevice() const { return m_Storage.Location() == Device; }
        
        const float* DataPtrUnsafe() const;
        const float* DeviceDataPtrUnsafe() const;

        const float* GetDevicePtr() const;
        float* GetDevicePtr();
        const cudnnTensorDescriptor_t DeviceDesc() const { return m_Shape.DeviceDesc(); }

        static TensorOpCpu* DefaultOp();
        static TensorOpCpu* ActiveOp();

        void DebugDumpValues(const string& outFile, bool includePtrs = false) const;
        void DebugRecoverValues(const string& inFile);

	private:
        TensorOpCpu* m_Op;
        Shape m_Shape;
        Storage m_Storage;
        string m_Name;

        TensorOpCpu* Op() const;

		static TensorOpCpu* GetOpFromMode(EOpMode mode);

		static TensorOpCpu* g_DefaultOp;
        static TensorOpCpu* g_ForcedOp;
		static TensorOpCpu* g_OpCpu;
        static TensorOpCpu* g_OpCpuMt;
        static TensorOpCpu* g_OpCpuMkl;
        static TensorOpCpu* g_OpGpu;

        friend class TensorOpGpu;
	};

    //////////////////////////////////////////////////////////////////////////
    _inline float Tensor::GetFlat(uint32_t i) const
    {
        CopyToHost();
        return m_Storage.Data()[i];
    }

    //////////////////////////////////////////////////////////////////////////
    _inline float& Tensor::Get(uint32_t w, uint32_t h, uint32_t d, uint32_t n)
    {
        CopyToHost(true);
        return m_Storage.Data()[m_Shape.GetIndex(w, h, d, n)];
    }

    //////////////////////////////////////////////////////////////////////////
    _inline float Tensor::Get(uint32_t w, uint32_t h, uint32_t d, uint32_t n) const
    {
        CopyToHost();
        return m_Storage.Data()[m_Shape.GetIndex(w, h, d, n)];
    }

    //////////////////////////////////////////////////////////////////////////
    _inline float& Tensor::operator()(uint32_t w, uint32_t h, uint32_t d, uint32_t n)
    {
        return Get(w, h, d, n);
    }

    //////////////////////////////////////////////////////////////////////////
    _inline float Tensor::operator()(uint32_t w, uint32_t h, uint32_t d, uint32_t n) const
    {
        return Get(w, h, d, n);
    }

    //////////////////////////////////////////////////////////////////////////
    _inline void Tensor::SetFlat(float value, uint32_t i)
    {
        CopyToHost(true);
        m_Storage.Data()[i] = value;
    }

    //////////////////////////////////////////////////////////////////////////
    _inline void Tensor::Set(float value, uint32_t w, uint32_t h, uint32_t d, uint32_t n)
    {
        CopyToHost(true);
        m_Storage.Data()[m_Shape.GetIndex(w, h, d, n)] = value;
    }

    NEURO_DLL_EXPORT Tensor operator*(const Tensor& t1, const Tensor& t2);
    NEURO_DLL_EXPORT Tensor operator*(const Tensor& t, float v);
    NEURO_DLL_EXPORT Tensor operator/(const Tensor& t1, const Tensor& t2);
    NEURO_DLL_EXPORT Tensor operator/(const Tensor& t, float v);
    NEURO_DLL_EXPORT Tensor operator/(float v, const Tensor& t);
    NEURO_DLL_EXPORT Tensor operator+(const Tensor& t1, const Tensor& t2);
    NEURO_DLL_EXPORT Tensor operator+(const Tensor& t, float v);
    NEURO_DLL_EXPORT Tensor operator-(const Tensor& t1, const Tensor& t2);
    NEURO_DLL_EXPORT Tensor operator-(const Tensor& t, float v);
    NEURO_DLL_EXPORT Tensor operator-(const Tensor& t);
    NEURO_DLL_EXPORT Tensor pow(const Tensor& t, float p);
    NEURO_DLL_EXPORT Tensor sqr(const Tensor& t);
    NEURO_DLL_EXPORT Tensor sqrt(const Tensor& t);
    NEURO_DLL_EXPORT Tensor sum(const Tensor& t, EAxis axis);
    NEURO_DLL_EXPORT Tensor mean(const Tensor& t, EAxis axis);

    NEURO_DLL_EXPORT Tensor zeros(const Shape& shape);
    NEURO_DLL_EXPORT Tensor ones(const Shape& shape);
}

#pragma warning(pop)
