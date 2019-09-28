#pragma once

#include "Types.h"

namespace Neuro
{
	class Tensor;

	struct ParameterAndGradient
	{
		ParameterAndGradient(Tensor* parameter, Tensor* gradient)
			: param(parameter), grad(gradient)
		{}

		Tensor* param;
		Tensor* grad;
	};

    struct SerializedParameter
    {
        SerializedParameter(Tensor* parameter, const vector<EAxis>& tranposeAxes = {})
            : param(parameter), transAxes(tranposeAxes)
        {}

        Tensor* param;
        vector<EAxis> transAxes;
    };
}