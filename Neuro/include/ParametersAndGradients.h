#pragma once

namespace Neuro
{
	class Tensor;

	struct ParametersAndGradients
	{
		ParametersAndGradients(Tensor* parameters, Tensor* gradients)
			: Parameters(parameters), Gradients(gradients)
		{}

		Tensor* Parameters;
		Tensor* Gradients;
	};
}