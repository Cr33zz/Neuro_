#pragma once

#include "Types.h"

namespace Neuro
{
	class Tensor;
    class Variable;

	struct ParameterAndGradient
	{
		ParameterAndGradient(Tensor* parameter, Tensor* gradient = nullptr)
			: param(parameter), grad(gradient)
		{}

		Tensor* param;
		Tensor* grad;
	};

    struct SerializedParameter
    {
        SerializedParameter(Variable* parameter, const vector<EAxis>& tranposeAxesKeras = {}, bool reshapeKeras = false)
            : param(parameter), transAxesKeras(tranposeAxesKeras), reshapeKeras(reshapeKeras)
        {}

        Variable* param;
        vector<EAxis> transAxesKeras;
        bool reshapeKeras;
    };
}