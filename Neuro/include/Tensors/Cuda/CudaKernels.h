#pragma once

namespace Neuro
{
    class Tensor;

	struct CudaKernels
	{
        static void Pad2D(const dim3& blocks, const dim3& threads, int outputLen, const float* inputDev, int inputStride1, int inputStride2, int inputStride3, int left, int right, int top, int bottom, float value, float* __restrict outputDev, int outputStride1, int outputStride2, int outputStride3);
        static void UpSample2D(const dim3& blocks, const dim3& threads, const float* inputDev, int inputWidth, int inputHeight, int inputDepth, int inputBatch, int scale, float* outputDev);
        static void UpSample2DGradient(const dim3& blocks, const dim3& threads, const float* outputGradientDev, int scale, float* inputGradientDev, int inputWidth, int inputHeight, int inputDepth, int inputBatch);
        static void LeakyReLU(const dim3& blocks, const dim3& threads, int inputLen, const float* inputDev, float alpha, float* outputDev);
        static void LeakyReLUGradient(const dim3& blocks, const dim3& threads, int inputLen, const float* outputDev, const float* outputGradientDev, float alpha, float* inputGradientDev);
        static void Pow(const dim3& blocks, const dim3& threads, int inputLen, const float* inputDev, float power, float* outputDev);
        static void PowGradient(const dim3& blocks, const dim3& threads, int inputLen, const float* inputDev, float power, const float* outputGradientDev, float* inputGradientDev);
        static void Abs(const dim3& blocks, const dim3& threads, int inputLen, const float* inputDev, float* outputDev);
        static void AbsGradient(const dim3& blocks, const dim3& threads, int inputLen, const float* inputDev, const float* outputGradientDev, float* inputGradientDev);
        static void Negate(const dim3& blocks, const dim3& threads, int inputLen, const float* inputDev, float* outputDev);
        static void Log(const dim3& blocks, const dim3& threads, int inputLen, const float* inputDev, float* outputDev);
        static void Inverse(const dim3& blocks, const dim3& threads, int inputLen, const float* inputDev, float alpha, float* outputDev);
        static void Add(const dim3& blocks, const dim3& threads, int inputLen, const float* inputDev, float v, float* outputDev, int subLen);
        // Assuming t1 has the same dimensions as output
        static void AddBroadcast(const dim3& blocks, const dim3& threads, float alpha, const float* t1Dev, float beta, const float* t2Dev, int t2Width, int t2Height, int t2Depth, int t2Batch, float* outputDev, int outputWidth, int outputHeight, int outputDepth, int outputBatch);
        static void Mul(const dim3& blocks, const dim3& threads, int len, const float* t1, const float* t2, float* outputDev, int subLen);
        // Assuming t1 has the same dimensions as output
        static void MulBroadcast(const dim3& blocks, const dim3& threads, float alpha, const float* t1Dev, float beta, const float* t2Dev, int t2Width, int t2Height, int t2Depth, int t2Batch, float* outputDev, int outputWidth, int outputHeight, int outputDepth, int outputBatch);
        static void Div(const dim3& blocks, const dim3& threads, int len, float alpha, const float* t1, float beta, const float* t2, float* outputDev);
        static void DivBroadcast(const dim3& blocks, const dim3& threads, float alpha, const float* t1Dev, int t1Width, int t1Height, int t1Depth, int t1Batch, float beta, const float* t2Dev, int t2Width, int t2Height, int t2Depth, int t2Batch, float* outputDev, int outputWidth, int outputHeight, int outputDepth, int outputBatch);
        static void Sum(const dim3& blocks, const dim3& threads, const float* inputDev, int inputWidth, int inputHeight, int inputDepth, int inputBatch, int axis, float* outputDev);
        static void AdamStep(const dim3& blocks, const dim3& threads, int inputLen, float* parameterDev, const float* gradientDev, float* mGradDev, float* vGradDev, float lr, float beta1, float beta2, float epsilon);
        static void SgdStep(const dim3& blocks, const dim3& threads, int inputLen, float* parameterDev, const float* gradientDev, float lr);
	};
}
