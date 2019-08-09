#pragma once

#include <functional>

#include "Types.h"

namespace Neuro
{
	using namespace std;

	class ActivationBase;
	class LayerBase;

	namespace TestTools
    {
        float DERIVATIVE_EPSILON = 1e-4f;
        float LOSS_DERIVATIVE_EPSILON = 1e-5f;

        bool ValidateLayer(LayerBase& layer);
        bool VerifyInputGradient(LayerBase& layer, int batchSize = 1);
        bool VerifyParametersGradient(LayerBase& layer, int batchSize = 1);
        tensor_ptr_vec_t GenerateInputsForLayer(LayerBase& layer, int batchSize);
        bool VerifyActivationFuncDerivative(ActivationBase& func, int batchSize = 1);
        bool VerifyLossFuncDerivative(LossBase& func, const Tensor& targetOutput, bool onlyPositiveOutput = false, int batchSize = 1, float tolerance = 0.01f);
        bool VerifyLossFunc(LossBase& func, const Tensor& targetOutput, function<float, float, float> testFunc, bool onlyPositiveOutput = false, int batchSize = 1);
    }
}
