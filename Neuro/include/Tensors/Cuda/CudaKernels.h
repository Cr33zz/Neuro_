#pragma once

namespace Neuro
{
    class Tensor;

	struct CudaKernels
	{
        static void One(const dim3& blocks, const dim3& threads, int inputLen, float* inputDev, int subLen);
        static void UpSample2D(const dim3& blocks, const dim3& threads, const float* inputDev, int inputWidth, int inputHeight, int inputDepth, int inputBatch, int scale, float* outputDev);
        static void UpSample2DGradient(const dim3& blocks, const dim3& threads, const float* outputGradientDev, int scale, float* inputGradientDev, int inputWidth, int inputHeight, int inputDepth, int inputBatch);
        static void LeakyReLU(const dim3& blocks, const dim3& threads, int inputLen, const float* inputDev, float alpha, float* outputDev);
        static void LeakyReLUGradient(const dim3& blocks, const dim3& threads, int inputLen, const float* outputDev, const float* outputGradientDev, float alpha, float* inputGradientDev);
        static void Mul(const dim3& blocks, const dim3& threads, int inputLen, const float* inputDev, float v, float* outputDev, int subLen);
        static void Div(const dim3& blocks, const dim3& threads, int inputLen, const float* inputDev, float v, float* outputDev, int subLen);
        static void Pow(const dim3& blocks, const dim3& threads, int inputLen, const float* inputDev, float power, float* outputDev, int subLen);
        static void PowGradient(const dim3& blocks, const dim3& threads, int inputLen, const float* inputDev, float power, const float* outputGradientDev, float* inputGradientDev, int subLen);
        static void Negate(const dim3& blocks, const dim3& threads, int inputLen, const float* inputDev, float* outputDev, int subLen);
        static void Inverse(const dim3& blocks, const dim3& threads, int inputLen, const float* inputDev, float* outputDev, int subLen);
        static void Add(const dim3& blocks, const dim3& threads, int inputLen, const float* inputDev, float v, float* outputDev, int subLen);
        // Assuming t1 has the same dimensions as output
        static void AddBroadcast(const dim3& blocks, const dim3& threads, float alpha, const float* t1Dev, float beta, const float* t2Dev, int t2Width, int t2Height, int t2Depth, int t2Batch, float* outputDev, int outputWidth, int outputHeight, int outputDepth, int outputBatch);
        static void MulElem(const dim3& blocks, const dim3& threads, int len, const float* t1, const float* t2, float* outputDev, int subLen);
        // Assuming t1 has the same dimensions as output
        static void MulElemBroadcast(const dim3& blocks, const dim3& threads, const float* t1Dev, const float* t2Dev, int t2Width, int t2Height, int t2Depth, int t2Batch, float* outputDev, int outputWidth, int outputHeight, int outputDepth, int outputBatch);
        static void Div(const dim3& blocks, const dim3& threads, int len, const float* t1, const float* t2, float* outputDev, int subLen);
        static void DivBroadcast(const dim3& blocks, const dim3& threads, const float* t1Dev, int t1Width, int t1Height, int t1Depth, int t1Batch, const float* t2Dev, int t2Width, int t2Height, int t2Depth, int t2Batch, float* outputDev, int outputWidth, int outputHeight, int outputDepth, int outputBatch);
        static void Sum(const dim3& blocks, const dim3& threads, const float* inputDev, int inputWidth, int inputHeight, int inputDepth, int inputBatch, int axis, float* outputDev);
        static void AdamStep(const dim3& blocks, const dim3& threads, int inputLen, float* parameterDev, const float* gradientDev, float* mGradDev, float* vGradDev, float batchSize, float lr, float beta1, float beta2, float epsilon);
        static void SgdStep(const dim3& blocks, const dim3& threads, int inputLen, float* parameterDev, const float* gradientDev, float batchSize, float lr);
	};
}
