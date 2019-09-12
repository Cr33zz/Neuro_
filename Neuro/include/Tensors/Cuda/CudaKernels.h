#pragma once

#include <nvfunctional>

namespace Neuro
{
    class Tensor;

	struct CudaKernels
	{
        static void Elu(const dim3& blocks, const dim3& threads, int inputLen, const float* inputDev, float alpha, float* outputDev);
        static void EluGradient(const dim3& blocks, const dim3& threads, int inputLen, const float* outputDev, const float* outputGradientDev, float alpha, float* inputGradientDev);
        static void Div(const dim3& blocks, const dim3& threads, int inputLen, const float* inputDev, float v, float* outputDev);
        static void Map(const dim3& blocks, const dim3& threads, int inputLen, const float* inputDev, const nvstd::function<float(float)>& func, float* outputDev);
	};
}
