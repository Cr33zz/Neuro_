#pragma once

#include <functional>

#include "Types.h"

namespace Neuro
{
	using namespace std;

	class ActivationBase;
	class LayerBase;
    class LossBase;

	namespace TestTools
    {
        extern float DERIVATIVE_EPSILON;
        extern float LOSS_DERIVATIVE_EPSILON;

        bool ValidateLayer(LayerBase& layer);
        bool VerifyInputGradient(LayerBase& layer, int batchSize = 1);
        bool VerifyParametersGradient(LayerBase& layer, int batchSize = 1);
        tensor_ptr_vec_t GenerateInputsForLayer(LayerBase& layer, int batchSize);
        bool VerifyActivationFuncDerivative(const ActivationBase& func, int batchSize = 1);
        bool VerifyLossFuncDerivative(const LossBase& func, const Tensor& targetOutput, bool onlyPositiveOutput = false, int batchSize = 1, float tolerance = 0.01f);
        template <typename F> bool VerifyLossFunc(const LossBase& func, const Tensor& targetOutput, F& testFunc, bool onlyPositiveOutput = false, int batchSize = 1);
    }
}
