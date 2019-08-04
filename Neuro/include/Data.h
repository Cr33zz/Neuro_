#pragma once

#include "Types.h"
#include "Tensors/Tensor.h"

namespace Neuro
{
	using namespace std;

	// Represents single input and corresponding output
    struct Data
    {
        Data(const tensor_ptr_vec_t& inputs, const tensor_ptr_vec_t& outputs)
			: Inputs(inputs), Outputs(outputs)
        {
        }

        Data(const Tensor* input, const Tensor* output)
			: Inputs(&input, &input), Outputs(&output, &output)
        {
        }

        const Tensor* Input() const { return Inputs[0]; }
        const Tensor* Output() const { return Outputs[0]; }

		tensor_ptr_vec_t Inputs;
		tensor_ptr_vec_t Outputs;
	};
}
