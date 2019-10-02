#pragma once

#include "Types.h"

namespace Neuro
{
	class Tensor;

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
        SerializedParameter(Tensor* parameter, const vector<EAxis>& tranposeAxesKeras = {})
            : param(parameter), transAxesKeras(tranposeAxesKeras)
        {}

        Tensor* param;
        vector<EAxis> transAxesKeras;
    };
}