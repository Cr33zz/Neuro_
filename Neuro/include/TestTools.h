#pragma once

#include <functional>
#include <string>

#include "Types.h"
#include "Stopwatch.h"
#include "Tensors/Tensor.h"

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

        bool ValidateLayer(LayerBase* layer);
        bool VerifyInputGradient(LayerBase* layer, int batchSize = 1);
        bool VerifyParametersGradient(LayerBase* layer, int batchSize = 1);
        tensor_ptr_vec_t GenerateInputsForLayer(LayerBase* layer, int batchSize);
        bool VerifyActivationFuncDerivative(const ActivationBase& func, int batchSize = 1, EOpMode mode = EOpMode::CPU);
        bool VerifyLossFuncDerivative(const LossBase& func, const Tensor& targetOutput, bool onlyPositiveOutput = false, int batchSize = 1, float tolerance = 0.01f);
        template <typename F> bool VerifyLossFunc(const LossBase& func, const Tensor& targetOutput, F& testFunc, bool onlyPositiveOutput = false, int batchSize = 1);

        class ProfileObj
        {
        public:
            ProfileObj();
            string ToString() const;

        private:
            Stopwatch m_Timer;
        };
    }
}

#define NEURO_CONCATENATE_DETAIL(x, y) x##y
#define NEURO_CONCATENATE(x, y) NEURO_CONCATENATE_DETAIL(x, y)
#define NEURO_UNIQUE_NAME(x) NEURO_CONCATENATE(x, __COUNTER__)
#define NEURO_PROFILE_INTERNAL(name, operation, var) \
Logger::WriteMessage(name##": "); \
TestTools::ProfileObj var; \
operation \
Logger::WriteMessage(var.ToString().c_str());
#define NEURO_PROFILE(name, operation) NEURO_PROFILE_INTERNAL(name, operation, NEURO_UNIQUE_NAME(p))
