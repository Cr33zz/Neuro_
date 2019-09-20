#pragma once

namespace Neuro
{
    class Tensor;

	struct CudaKernels
	{
        static void Elu(const dim3& blocks, const dim3& threads, int inputLen, const float* inputDev, float alpha, float* outputDev);
        static void EluGradient(const dim3& blocks, const dim3& threads, int inputLen, const float* outputDev, const float* outputGradientDev, float alpha, float* inputGradientDev);
        static void LeakyReLU(const dim3& blocks, const dim3& threads, int inputLen, const float* inputDev, float alpha, float* outputDev);
        static void LeakyReLUGradient(const dim3& blocks, const dim3& threads, int inputLen, const float* outputDev, const float* outputGradientDev, float alpha, float* inputGradientDev);
        static void Div(const dim3& blocks, const dim3& threads, int inputLen, const float* inputDev, float v, float* outputDev);
        static void AddBroadcast(const dim3& blocks, const dim3& threads, float alpha, const float* t1Dev, int t1Width, int t1Height, int t1Depth, int t1Batch, float beta, const float* t2Dev, int t2Width, int t2Height, int t2Depth, int t2Batch, float* outputDev, int outputWidth, int outputHeight, int outputDepth, int outputBatch);
        static void AdamStep(const dim3& blocks, const dim3& threads, int inputLen, float* parameterDev, float* gradientDev, float* mGradDev, float* vGradDev, float batchSize, float lr, float beta1, float beta2, float epsilon);
        static void SgdStep(const dim3& blocks, const dim3& threads, int inputLen, float* parameterDev, float* gradientDev, float batchSize, float lr);
	};
}
