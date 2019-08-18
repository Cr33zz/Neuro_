#include "TestTools.h"
#include "Tensors/Tensor.h"
#include "Layers/LayerBase.h"
#include "Activations.h"
#include "Loss.h"

namespace Neuro
{
    float TestTools::DERIVATIVE_EPSILON = 1e-4f;
    float TestTools::LOSS_DERIVATIVE_EPSILON = 1e-5f;

	//////////////////////////////////////////////////////////////////////////
	bool TestTools::ValidateLayer(LayerBase* layer)
	{
		return VerifyInputGradient(layer) && 
			   VerifyInputGradient(layer, 3) &&
			   VerifyParametersGradient(layer) && 
               VerifyParametersGradient(layer, 3);
	}

	//////////////////////////////////////////////////////////////////////////
	bool TestTools::VerifyInputGradient(LayerBase* layer, int batchSize)
	{
		Tensor::SetDefaultOpMode(Tensor::EOpMode::CPU);
		auto inputs = GenerateInputsForLayer(layer, batchSize);

		auto output = layer->FeedForward(inputs, true);
		auto outputGradient = Tensor(output->GetShape());
		outputGradient.FillWithValue(1);

		layer->BackProp(outputGradient);

		auto result = Tensor(output->GetShape());

		for (int n = 0; n < (int)inputs.size(); ++n)
		{
			auto& input = const_cast<Tensor&>(*inputs[n]);
			for (int i = 0; i < input.GetShape().Length; ++i)
			{
				result.Zero();

				auto oldValue = input.GetFlat(i);

				input.SetFlat(oldValue - DERIVATIVE_EPSILON, i);
				auto output1 = *layer->FeedForward(inputs, true);
				input.SetFlat(oldValue + DERIVATIVE_EPSILON, i);
				auto output2 = *layer->FeedForward(inputs, true);

				input.SetFlat(oldValue, i);

				output2.Sub(output1, result);

				vector<float> approxGrad(output->GetShape().Length);
				float approxGradient = 0;
				for (int j = 0; j < output->GetShape().Length; j++)
				{
					approxGrad[j] = result.GetFlat(j) / (2.0f * DERIVATIVE_EPSILON);
					approxGradient += approxGrad[j];
				}

				if (abs(approxGradient - layer->InputsGradient()[n].GetFlat(i)) > 0.02f)
				{
					//Assert::Fail(string("Input gradient validation failed at element ") + to_string(i) + " of input " + to_string(n) + ", expected " + to_string(approxGradient) + " actual " + to_string(layer->InputsGradient[n].GetFlat(i)) + "!");
					return false;
				}
			}
		}

		return true;
	}

	//////////////////////////////////////////////////////////////////////////
	bool TestTools::VerifyParametersGradient(LayerBase* layer, int batchSize)
	{
        Tensor::SetDefaultOpMode(Tensor::EOpMode::CPU);
        auto inputs = GenerateInputsForLayer(layer, batchSize);

        auto output = layer->FeedForward(inputs, true);
        auto outputGradient = Tensor(output->GetShape());
        outputGradient.FillWithValue(1);

        layer->BackProp(outputGradient);

        vector<ParametersAndGradients> paramsAndGrads;
		layer->GetParametersAndGradients(paramsAndGrads);

		if (paramsAndGrads.empty())
			return true;

        auto result = Tensor(output->GetShape());

		auto parameters = paramsAndGrads[0].Parameters;
		auto gradients = paramsAndGrads[0].Gradients;

		for (int i = 0; i < parameters->GetShape().Length; i++)
		{
			result.Zero();

			float oldValue = parameters->GetFlat(i);
			parameters->SetFlat(oldValue + DERIVATIVE_EPSILON, i);
			auto output1 = Tensor(*layer->FeedForward(inputs, true));
			parameters->SetFlat(oldValue - DERIVATIVE_EPSILON, i);
			auto output2 = Tensor(*layer->FeedForward(inputs, true));

			parameters->SetFlat(oldValue, i);

			output1.Sub(output2, result);

            vector<float> approxGrad(output->GetShape().Length);
            float approxGradient = 0;
            for (int j = 0; j < output->GetShape().Length; j++)
            {
                approxGrad[j] = result.GetFlat(j) / (2.0f * DERIVATIVE_EPSILON);
                approxGradient += approxGrad[j];
            }

			if (abs(approxGradient - gradients->GetFlat(i)) > 0.02)
			{
				//Debug.Assert(false, $"Parameter gradient validation failed at parameter {i}, expected {approxGradient} actual {gradients.GetFlat(i)}!");
				return false;
			}
		}

		return true;
	}

	//////////////////////////////////////////////////////////////////////////
	Neuro::tensor_ptr_vec_t TestTools::GenerateInputsForLayer(LayerBase* layer, int batchSize)
	{
		tensor_ptr_vec_t inputs(layer->InputShapes().size());

		for (int i = 0; i < (int)inputs.size(); ++i)
		{
			auto input = new Tensor(Shape(layer->InputShape().Width(), layer->InputShape().Height(), layer->InputShape().Depth(), batchSize));
			input->FillWithRand(7 + i);
			inputs[i] = input;
		}

		return inputs;
	}

	//////////////////////////////////////////////////////////////////////////
	bool TestTools::VerifyActivationFuncDerivative(const ActivationBase& func, int batchSize, Tensor::EOpMode mode)
	{
        Tensor::SetDefaultOpMode(mode);
		auto input = Tensor(Shape(3, 3, 3, batchSize));
		input.FillWithRange(-1.0f, 2.0f / input.Length());

		auto outputGradient = Tensor(Shape(3, 3, 3, batchSize));
		outputGradient.FillWithValue(1.0f);

		// for derivation purposes activation functions expect already processed input
		auto output = Tensor(input.GetShape());
		func.Compute(input, output);

		auto derivative = Tensor(input.GetShape());
		func.Derivative(output, outputGradient, derivative);

		auto output1 = Tensor(input.GetShape());
		func.Compute(input.Sub(DERIVATIVE_EPSILON), output1);

		auto output2 = Tensor(input.GetShape());
		func.Compute(input.Add(DERIVATIVE_EPSILON), output2);

		auto result = Tensor(input.GetShape());
		output2.Sub(output1, result);

		auto approxDerivative = result.Div(2 * DERIVATIVE_EPSILON);

		return approxDerivative.Equals(derivative, 1e-2f);
	}

	//////////////////////////////////////////////////////////////////////////
	bool TestTools::VerifyLossFuncDerivative(const LossBase& func, const Tensor& targetOutput, bool onlyPositiveOutput, int batchSize, float tolerance)
	{
        Tensor::SetDefaultOpMode(Tensor::EOpMode::CPU);
        auto output = Tensor(Shape(3, 3, 3, batchSize));
		output.FillWithRand(10, onlyPositiveOutput ? 0.f : -1.f);

		// for derivation purposes activation functions expect already processed input
		auto error = Tensor(output.GetShape());
		func.Compute(targetOutput, output, error);

		auto derivative = Tensor(output.GetShape());
		func.Derivative(targetOutput, output, derivative);

		auto error1 = Tensor(output.GetShape());
		func.Compute(targetOutput, output.Sub(LOSS_DERIVATIVE_EPSILON), error1);

		auto error2 = Tensor(output.GetShape());
		func.Compute(targetOutput, output.Add(LOSS_DERIVATIVE_EPSILON), error2);

		auto result = Tensor(output.GetShape());
		error2.Sub(error1, result);

		auto approxDerivative = result.Div(2 * LOSS_DERIVATIVE_EPSILON);

		return approxDerivative.Equals(derivative, tolerance);
	}

	//////////////////////////////////////////////////////////////////////////
    template <typename F>
	bool TestTools::VerifyLossFunc(const LossBase& func, const Tensor& targetOutput, F& testFunc, bool onlyPositiveOutput, int batchSize)
	{
        Tensor::SetDefaultOpMode(Tensor::EOpMode::CPU);
        auto output = Tensor(Shape(3, 3, 3, batchSize));
		output.FillWithRand(10, onlyPositiveOutput ? 0 : -1);

		auto error = new Tensor(output.Shape);
		func.Compute(targetOutput, output, error);

		for (int i = 0; i < output.Shape.Length; ++i)
		{
			if (abs(error.GetFlat(i) - testFunc(targetOutput.GetFlat(i), output.GetFlat(i))) > 1e-4)
				return false;
		}

		return true;
	}
}
