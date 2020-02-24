#pragma once

namespace Neuro
{
    class Tensor;

	struct CudaKernels
	{
        static void ExtractSubTensor2D(const dim3& blocks, const dim3& threads, int outputLen, const float* inputDev, int inputStride1, int inputStride2, int inputStride3, int widthOffset, int heightOffset, float* __restrict outputDev, int outputStride1, int outputStride2, int outputStride3);
        static void FuseSubTensor2D(const dim3& blocks, const dim3& threads, int inputLen, const float* inputDev, int inputStride1, int inputStride2, int inputStride3, int widthOffset, int heightOffset, bool add, float* __restrict outputDev, int outputStride1, int outputStride2, int outputStride3);
        static void ConstantPad2D(const dim3& blocks, const dim3& threads, int outputLen, const float* inputDev, int inputStride1, int inputStride2, int inputStride3, int left, int right, int top, int bottom, float value, float* __restrict outputDev, int outputStride1, int outputStride2, int outputStride3);
        static void ReflectPad2D(const dim3& blocks, const dim3& threads, int outputLen, const float* inputDev, int inputStride1, int inputStride2, int inputStride3, int left, int right, int top, int bottom, float* __restrict outputDev, int outputStride1, int outputStride2, int outputStride3);
        static void Pad2DGradient(const dim3& blocks, const dim3& threads, int inputGradLen, const float* outputGradDev, int outputGradStride1, int outputGradStride2, int outputGradStride3, int left, int right, int top, int bottom, float* inputGradDev, int inputGradStride1, int inputGradStride2, int inputGradStride3);
        static void UpSample2D(const dim3& blocks, const dim3& threads, const float* inputDev, int inputWidth, int inputHeight, int inputDepth, int inputBatch, int scale, float* outputDev);
        static void UpSample2DGradient(const dim3& blocks, const dim3& threads, const float* outputGradientDev, int scale, float* inputGradientDev, int inputWidth, int inputHeight, int inputDepth, int inputBatch);
        static void LeakyReLU(const dim3& blocks, const dim3& threads, int inputLen, const float* inputDev, float alpha, float* outputDev);
        static void LeakyReLUGradient(const dim3& blocks, const dim3& threads, int inputLen, const float* outputDev, const float* outputGradientDev, float alpha, float* inputGradientDev);
        static void Pow(const dim3& blocks, const dim3& threads, int inputLen, const float* inputDev, float power, float* outputDev);
        static void PowGradient(const dim3& blocks, const dim3& threads, int inputLen, const float* inputDev, float power, const float* outputGradientDev, float* inputGradientDev);
        static void Clip(const dim3& blocks, const dim3& threads, int inputLen, const float* inputDev, float min, float max, float* outputDev);
        static void ClipGradient(const dim3& blocks, const dim3& threads, int inputLen, const float* inputDev, float min, float max, const float* outputGradientDev, float* inputGradientDev);
        static void Abs(const dim3& blocks, const dim3& threads, int inputLen, const float* inputDev, float* outputDev);
        static void AbsGradient(const dim3& blocks, const dim3& threads, int inputLen, const float* inputDev, const float* outputGradientDev, float* inputGradientDev);
        static void Negate(const dim3& blocks, const dim3& threads, int inputLen, const float* inputDev, float* outputDev);
        static void Log(const dim3& blocks, const dim3& threads, int inputLen, const float* inputDev, float* outputDev);
        static void Inverse(const dim3& blocks, const dim3& threads, int inputLen, const float* inputDev, float alpha, float* outputDev);
        static void Add(const dim3& blocks, const dim3& threads, int inputLen, const float* inputDev, float v, float* outputDev);
        static void AddBroadcast(const dim3& blocks, const dim3& threads, float alpha, const float* t1Dev, int t1Width, int t1Height, int t1Depth, int t1Batch, float beta, const float* t2Dev, int t2Width, int t2Height, int t2Depth, int t2Batch, float* outputDev, int outputWidth, int outputHeight, int outputDepth, int outputBatch);
        static void Mul(const dim3& blocks, const dim3& threads, int len, const float* t1, const float* t2, float* outputDev, int subLen);
        static void MulBroadcast(const dim3& blocks, const dim3& threads, float alpha, const float* t1Dev, int t1Width, int t1Height, int t1Depth, int t1Batch, float beta, const float* t2Dev, int t2Width, int t2Height, int t2Depth, int t2Batch, float* outputDev, int outputWidth, int outputHeight, int outputDepth, int outputBatch);
        static void Div(const dim3& blocks, const dim3& threads, int len, float alpha, const float* t1, float beta, const float* t2, float* outputDev);
        static void DivBroadcast(const dim3& blocks, const dim3& threads, float alpha, const float* t1Dev, int t1Width, int t1Height, int t1Depth, int t1Batch, float beta, const float* t2Dev, int t2Width, int t2Height, int t2Depth, int t2Batch, float* outputDev, int outputWidth, int outputHeight, int outputDepth, int outputBatch);
        static void Sum(const dim3& blocks, const dim3& threads, const float* inputDev, int inputWidth, int inputHeight, int inputDepth, int inputBatch, int axis, float* outputDev);
        static void AdamStep(const dim3& blocks, const dim3& threads, int inputLen, float* parameterDev, const float* gradientDev, float* mGradDev, float* vGradDev, float lr, float beta1, float beta2, float epsilon);
        static void SgdStep(const dim3& blocks, const dim3& threads, int inputLen, float* parameterDev, const float* gradientDev, float lr);
        static void Dropout(const dim3& blocks, const dim3& threads, int inputLen, const float* inputDev, float prob, float* maskDev, float* outputDev);
        static void DropoutGradient(const dim3& blocks, const dim3& threads, int inputLen, const float* outputGradDev, const float* maskDev, float* inputGradDev);
        static void Transpose(const dim3& blocks, const dim3& threads, int inputLen, const float* inputDev, int axis0, int axis1, int axis2, int axis3, int stride0, int stride1, int stride2, int stride3, float* outputDev, int outputStride1, int outputStride2, int outputStride3);
	};
}
