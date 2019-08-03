#pragma once

#include <vector>
#include "Tensors/Tensor.h"

namespace Neuro
{
	using namespace std;

    struct Data
    {
        Data(const vector<Tensor>& inputs, const vector<Tensor>& outputs)
			: Inputs(inputs), Outputs(outputs)
        {
        }

        Data(Tensor input, Tensor output)
			: Inputs(&input, &input), Outputs(&output, &output)
        {
        }

        const vector<Tensor> Inputs;
		const vector<Tensor> Outputs;

        const Tensor& Input() const { return Inputs[0]; }
        const Tensor& Output() const { return Outputs[0]; }
	};
}
