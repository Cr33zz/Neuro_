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
        static void AdamStep(const dim3& blocks, const dim3& threads, int inputLen, float* parameterDev, float* gradientDev, float* mGradDev, float* vGradDev, float batchSize, float lr, float beta1, float beta2, float epsilon);
        static void SgdStep(const dim3& blocks, const dim3& threads, int inputLen, float* parameterDev, float* gradientDev, float batchSize, float lr);
	};
}
