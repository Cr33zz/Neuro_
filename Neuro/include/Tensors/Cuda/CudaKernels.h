#pragma once

namespace Neuro
{
    class Tensor;

	struct CudaKernels
	{
        static void Elu(const dim3& blocks, const dim3& threads, int inputLen, const float* inputDev, float alpha, float* resultDev);
        static void EluGradient(const dim3& blocks, const dim3& threads, int inputLen, const float* outputDev, const float* outputGradientDev, float alpha, float* resultDev);
	};
}